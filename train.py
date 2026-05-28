import os
import re
import json
import shutil
import glob as glob_module
import argparse
import logging
import math
from logging.handlers import RotatingFileHandler
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from easydict import EasyDict

from AbMEGD.datasets import get_dataset
from AbMEGD.models import get_model
from AbMEGD.utils.misc import *
from AbMEGD.utils.data import *
from AbMEGD.utils.train import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================================================
# Logging
# =========================================================
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            pass


class RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


def setup_logger(run_dir: str, rank: int, debug: bool = False):
    """
    logger      : terminal + file
    file_logger : file only (ABX-style detailed logging)
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    rank_filter = RankFilter(rank)

    # terminal: debug时显示INFO；正常训练时只显示WARNING/ERROR，避免刷屏
    term_handler = TqdmLoggingHandler()
    term_handler.setLevel(logging.INFO if debug else logging.WARNING)
    term_handler.addFilter(rank_filter)
    term_handler.setFormatter(
        logging.Formatter("%(asctime)s [R%(rank)s] %(levelname)s %(message)s", "%H:%M:%S")
    )
    logger.addHandler(term_handler)

    file_logger = logging.getLogger("file_only")
    file_logger.handlers.clear()
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False

    if (rank in (-1, 0)) and (not debug):
        os.makedirs(run_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=os.path.join(run_dir, "train.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(rank_filter)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        file_logger.addHandler(file_handler)
    else:
        file_logger.addHandler(logging.NullHandler())

    # 压低 distributed 自己的 INFO 噪音
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)

    return logger, file_logger


# =========================================================
# EMA
# =========================================================
class EMA:
    """
    Exponential Moving Average for model parameters.
    Supports bias-corrected early updates.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, use_num_updates: bool = True):
        self.model = model
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.num_updates = 0 if use_num_updates else None
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self):
        decay = self.decay
        if self.use_num_updates and self.num_updates is not None:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
                continue
            self.shadow[name] = decay * self.shadow[name].to(p.device) + one_minus_decay * p.detach()

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.device))

    @torch.no_grad()
    def restore(self):
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict) or "shadow" not in state_dict:
            logging.warning("[EMA] Loading legacy EMA state (shadow only).")
            shadow_state = state_dict
            self.num_updates = None
        else:
            shadow_state = state_dict["shadow"]
            self.decay = state_dict.get("decay", self.decay)
            self.num_updates = state_dict.get("num_updates", None)

        model_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
        for k in model_keys:
            if k in shadow_state:
                self.shadow[k] = shadow_state[k].clone()

    @torch.no_grad()
    def get_ema_model_state(self):
        """
        Return EMA-applied state_dict without modifying the live model permanently.
        """
        current = {}
        ema_state = {}
        for name, p in self.model.named_parameters():
            current[name] = p.detach().clone()
            if p.requires_grad and name in self.shadow:
                ema_state[name] = self.shadow[name].to(p.device).clone()

        # Temporarily apply EMA
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in ema_state:
                p.data.copy_(ema_state[name])

        state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

        # Restore original live weights
        for name, p in self.model.named_parameters():
            if name in current:
                p.data.copy_(current[name])

        return state


# =========================================================
# DDP helpers
# =========================================================
def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank):
    return rank in (-1, 0)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, local_rank, world_size
    return False, 0, -1, 1


def ddp_cleanup():
    if is_dist():
        dist.destroy_process_group()


def reduce_scalar(x: float, device, op=dist.ReduceOp.SUM):
    if not is_dist():
        return float(x)
    t = torch.tensor([x], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=op)
    return float(t.item())


def reduce_dict_mean(d: dict, device):
    if not is_dist():
        return {k: float(v) for k, v in d.items()}
    keys = sorted(d.keys())
    if len(keys) == 0:
        return {}
    vals = torch.tensor([float(d[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals = vals / dist.get_world_size()
    return {k: float(vals[i].item()) for i, k in enumerate(keys)}


# =========================================================
# Data helpers
# =========================================================
def infer_batch_size(batch):
    if isinstance(batch, dict):
        for v in batch.values():
            bs = infer_batch_size(v)
            if bs is not None:
                return bs
    elif isinstance(batch, (list, tuple)):
        for v in batch:
            bs = infer_batch_size(v)
            if bs is not None:
                return bs
    elif torch.is_tensor(batch) and batch.dim() > 0:
        return int(batch.shape[0])
    return None


def infinite_loader(loader, sampler=None, start_epoch=0):
    epoch = start_epoch
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def resolve_training_schedule(
    train_dataset_size: int,
    batch_size_per_gpu: int,
    accumulation_steps: int,
    world_size: int,
    train_epochs: int,
):
    """
    ABX-style logic:
      effective_global_batch = per_gpu_batch * world_size * accumulation_steps
      updates_per_epoch = ceil(num_train_samples / effective_global_batch)
      max_global_steps = train_epochs * updates_per_epoch
    """
    if train_dataset_size <= 0:
        raise ValueError("train_dataset_size must be positive")
    if batch_size_per_gpu <= 0:
        raise ValueError("batch_size_per_gpu must be positive")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if train_epochs <= 0:
        raise ValueError("train_epochs must be positive")

    effective_global_batch = batch_size_per_gpu * accumulation_steps * world_size
    updates_per_epoch = math.ceil(train_dataset_size / effective_global_batch)
    max_global_steps = train_epochs * updates_per_epoch

    return {
        "train_dataset_size": train_dataset_size,
        "batch_size_per_gpu": batch_size_per_gpu,
        "accumulation_steps": accumulation_steps,
        "world_size": world_size,
        "effective_global_batch": effective_global_batch,
        "updates_per_epoch": updates_per_epoch,
        "train_epochs": train_epochs,
        "max_global_steps": max_global_steps,
    }


def resolve_max_global_steps(
    target_total_samples: int,
    batch_size_per_gpu: int,
    accumulation_steps: int,
    world_size: int,
):
    """
    Keep TOTAL seen samples fixed, regardless of:
      - batch_size_per_gpu
      - accumulation_steps
      - world_size

    max_global_steps = ceil(target_total_samples / effective_global_batch)
    """
    if target_total_samples <= 0:
        raise ValueError("target_total_samples must be positive")
    if batch_size_per_gpu <= 0:
        raise ValueError("batch_size_per_gpu must be positive")
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    effective_global_batch = (
        batch_size_per_gpu * world_size * accumulation_steps
    )
    max_global_steps = math.ceil(target_total_samples / effective_global_batch)

    return {
        "target_total_samples": target_total_samples,
        "batch_size_per_gpu": batch_size_per_gpu,
        "accumulation_steps": accumulation_steps,
        "world_size": world_size,
        "effective_global_batch": effective_global_batch,
        "max_global_steps": max_global_steps,
    }
        
# =========================================================
# Config defaults / runtime overrides
# =========================================================
def ensure_config_sections(config):
    if not hasattr(config, "train") or config.train is None:
        config.train = EasyDict()

    if not hasattr(config.train, "loss_weights") or config.train.loss_weights is None:
        config.train.loss_weights = EasyDict({"rot": 1.0, "pos": 1.0, "seq": 1.0})

    if not hasattr(config.train, "seed"):
        config.train.seed = 2024
    if not hasattr(config.train, "batch_size"):
        config.train.batch_size = 2
    if not hasattr(config.train, "target_total_samples"):
        config.train.target_total_samples = 3200000
    if not hasattr(config.train, "max_iters"):
        config.train.max_iters = 200000
    if not hasattr(config.train, "val_freq"):
        config.train.val_freq = 1000
    if not hasattr(config.train, "max_grad_norm"):
        config.train.max_grad_norm = 100.0

    if not hasattr(config.train, "optimizer") or config.train.optimizer is None:
        config.train.optimizer = EasyDict()
    if not hasattr(config.train.optimizer, "type"):
        config.train.optimizer.type = "adam"
    if not hasattr(config.train.optimizer, "lr"):
        config.train.optimizer.lr = 1e-4
    if not hasattr(config.train.optimizer, "weight_decay"):
        config.train.optimizer.weight_decay = 0.0
    if not hasattr(config.train.optimizer, "beta1"):
        config.train.optimizer.beta1 = 0.9
    if not hasattr(config.train.optimizer, "beta2"):
        config.train.optimizer.beta2 = 0.999

    if not hasattr(config.train, "scheduler") or config.train.scheduler is None:
        config.train.scheduler = EasyDict()
    if not hasattr(config.train.scheduler, "type"):
        config.train.scheduler.type = "plateau"
    if not hasattr(config.train.scheduler, "factor"):
        config.train.scheduler.factor = 0.8
    if not hasattr(config.train.scheduler, "patience"):
        config.train.scheduler.patience = 10
    if not hasattr(config.train.scheduler, "min_lr"):
        config.train.scheduler.min_lr = 5e-6

    return config

def apply_runtime_overrides(config, args):
    if args.seed is not None:
        config.train.seed = int(args.seed)

    if args.batch_size is not None:
        config.train.batch_size = int(args.batch_size)

    if args.target_total_samples is not None:
        config.train.target_total_samples = int(args.target_total_samples)
        
    if args.max_iters is not None:
        config.train.max_iters = int(args.max_iters)

    if args.val_freq is not None:
        config.train.val_freq = int(args.val_freq)

    if args.max_grad_norm is not None:
        config.train.max_grad_norm = float(args.max_grad_norm)

    if args.lr is not None:
        config.train.optimizer.lr = float(args.lr)

    if args.weight_decay is not None:
        config.train.optimizer.weight_decay = float(args.weight_decay)

    if args.beta1 is not None:
        config.train.optimizer.beta1 = float(args.beta1)

    if args.beta2 is not None:
        config.train.optimizer.beta2 = float(args.beta2)

    if args.scheduler_factor is not None:
        config.train.scheduler.factor = float(args.scheduler_factor)

    if args.scheduler_patience is not None:
        config.train.scheduler.patience = int(args.scheduler_patience)

    if args.scheduler_min_lr is not None:
        config.train.scheduler.min_lr = float(args.scheduler_min_lr)

    return config

def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    return obj


# =========================================================
# Snapshot / run_dir
# =========================================================
def resolve_run_dir(args, config_name, rank):
    """
    DDP-safe run_dir creation:
      - resume: all ranks infer from resume path
      - fresh run: only rank0 creates new log dir, then broadcast
    """
    if args.resume:
        run_dir = os.path.abspath(os.path.join(os.path.dirname(args.resume), ".."))
        return run_dir

    run_dir = None
    if is_main_process(rank):
        run_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)

    if is_dist():
        obj_list = [run_dir]
        dist.broadcast_object_list(obj_list, src=0)
        run_dir = obj_list[0]

    return run_dir


def snapshot_configs(run_dir, args, config=None):
    snap_dir = os.path.join(run_dir, "config_snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    try:
        if os.path.isfile(args.config):
            shutil.copy2(args.config, os.path.join(snap_dir, os.path.basename(args.config)))
    except Exception as e:
        logging.warning(f"[Snapshot] failed to copy config file: {e}")

    try:
        with open(os.path.join(run_dir, "runtime_args.json"), "w", encoding="utf-8") as fw:
            json.dump(vars(args), fw, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"[Snapshot] failed to write runtime_args.json: {e}")

    if config is not None:
        try:
            with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as fw:
                json.dump(_to_plain_dict(config), fw, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"[Snapshot] failed to write resolved_config.json: {e}")


# =========================================================
# Checkpoint manager
# =========================================================
class CheckpointManager:
    def __init__(self, ckpt_dir, best_keys):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_keys = ["overall"] + [k for k in best_keys if k != "overall"]
        self.best_metrics = {k: float("inf") for k in self.best_keys}
        self.prev_best_paths = {k: None for k in self.best_keys}
        self.prev_last_path = None

        self.sync_from_disk()

    @staticmethod
    def _sanitize_metric_name(name: str):
        return re.sub(r"[^0-9a-zA-Z_]+", "_", name)

    @staticmethod
    def _parse_step_loss(path):
        m = re.search(r"step(\d+)_loss([0-9]+(?:\.[0-9]+)?)\.pt$", os.path.basename(path))
        if not m:
            return None
        return int(m.group(1)), float(m.group(2))

    def sync_from_disk(self):
        last_ckpts = glob_module.glob(os.path.join(self.ckpt_dir, "last_step*.pt"))
        if last_ckpts:
            parsed = []
            for p in last_ckpts:
                info = self._parse_step_loss(p)
                if info is not None:
                    parsed.append((info[0], p))
            if parsed:
                parsed.sort(key=lambda x: x[0])
                self.prev_last_path = parsed[-1][1]

        for key in self.best_keys:
            skey = self._sanitize_metric_name(key)
            ckpts = glob_module.glob(os.path.join(self.ckpt_dir, f"best_{skey}_step*.pt"))
            parsed = []
            for p in ckpts:
                info = self._parse_step_loss(p)
                if info is not None:
                    parsed.append((info[1], p))
            if parsed:
                parsed.sort(key=lambda x: x[0])
                self.best_metrics[key] = parsed[0][0]
                self.prev_best_paths[key] = parsed[0][1]

    def _safe_remove(self, path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def save(
        self,
        tag,
        metric_value,
        model,
        optimizer,
        scheduler,
        ema,
        config,
        args,
        iteration,
        global_step,
        extra_state=None,
    ):
        target_model = unwrap_model(model)
        raw_model_state = {k: v.detach().clone() for k, v in target_model.state_dict().items()}
        export_model_state = ema.get_ema_model_state() if ema is not None else raw_model_state

        payload = {
            "config": config,
            "args": vars(args),
            "iteration": iteration,
            "global_step": global_step,
            "metric": float(metric_value),
            "model": export_model_state,   # inference-friendly (EMA if enabled)
            "model_raw": raw_model_state,  # resume-friendly
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "ema": ema.state_dict() if ema is not None else None,
            "best_metrics": self.best_metrics,
        }
        if extra_state is not None:
            payload.update(extra_state)

        if tag == "last":
            self._safe_remove(self.prev_last_path)
            fname = f"last_step{global_step}_loss{metric_value:.4f}.pt"
            path = os.path.join(self.ckpt_dir, fname)
            torch.save(payload, path)
            self.prev_last_path = path
            return path

        if tag.startswith("best_"):
            metric_key = tag[len("best_"):]
            if metric_key not in self.best_metrics:
                raise ValueError(f"Unknown best metric key: {metric_key}")
            self._safe_remove(self.prev_best_paths[metric_key])
            skey = self._sanitize_metric_name(metric_key)
            fname = f"best_{skey}_step{global_step}_loss{metric_value:.4f}.pt"
            path = os.path.join(self.ckpt_dir, fname)
            torch.save(payload, path)
            self.prev_best_paths[metric_key] = path
            self.best_metrics[metric_key] = float(metric_value)
            return path

        raise ValueError(f"Unknown checkpoint tag: {tag}")


# =========================================================
# Validation
# =========================================================
@torch.no_grad()
def validate(global_step, model, ema, val_loader, config, args, writer, rank, device):
    target_model = unwrap_model(model)

    if ema is not None:
        ema.apply_shadow()

    target_model.eval()

    loss_sums = {}
    sample_count = 0

    for batch in val_loader:
        batch = recursive_to(batch, args.device)

        loss_dict = target_model(batch)
        overall = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict["overall"] = overall

        bs = infer_batch_size(batch)
        if bs is None:
            bs = 1

        sample_count += bs
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().item()) * bs
            else:
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v) * bs

    if is_dist():
        count_tensor = torch.tensor([sample_count], device=device, dtype=torch.float32)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_count = max(int(count_tensor.item()), 1)

        keys = sorted(loss_sums.keys())
        if len(keys) > 0:
            sums_tensor = torch.tensor([loss_sums[k] for k in keys], device=device, dtype=torch.float32)
            dist.all_reduce(sums_tensor, op=dist.ReduceOp.SUM)
            avg_dict = {k: float(sums_tensor[i].item()) / global_count for i, k in enumerate(keys)}
        else:
            avg_dict = {}
    else:
        global_count = max(sample_count, 1)
        avg_dict = {k: float(v) / global_count for k, v in loss_sums.items()}

    if ema is not None:
        ema.restore()

    target_model.train()

    if is_main_process(rank) and writer is not None:
        for k, v in avg_dict.items():
            writer.add_scalar(f"LossVal/{k}", v, global_step)

    return avg_dict


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation micro-steps per optimizer update")
    parser.add_argument("--log_steps", type=int, default=50,
                        help="Write train loss to file every N global steps")

    # runtime overrides for frequently changed knobs
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=200000)
    parser.add_argument("--target_total_samples", type=int, default=3200000)
    parser.add_argument("--val_freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    parser.add_argument("--scheduler_factor", type=float, default=None)
    parser.add_argument("--scheduler_patience", type=int, default=None)
    parser.add_argument("--scheduler_min_lr", type=float, default=None)

    # EMA / checkpoint
    parser.add_argument("--use_ema", action="store_true", default=False, help="Enable EMA")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--save_best_keys", type=str, nargs="*", default=None,
                        help="Additional validation metrics to save best checkpoints for, e.g. seq pos rot")
    parser.add_argument("--save_interval", type=int, default=0,
                        help="Optional save-last interval in global steps (0 means only save last at validation)")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", default=False,
                        help="Set find_unused_parameters=True in DDP")

    args = parser.parse_args()
    assert isinstance(args.accumulation_steps, int) and args.accumulation_steps >= 1

    # -----------------------------------------------------
    # DDP setup
    # -----------------------------------------------------
    is_ddp, rank, local_rank, world_size = ddp_setup()
    if local_rank != -1:
        args.device = f"cuda:{local_rank}"

    # -----------------------------------------------------
    # Config / seed / run dir
    # -----------------------------------------------------
    config, config_name = load_config(args.config)
    config = ensure_config_sections(config)
    config = apply_runtime_overrides(config, args)

    seed_base = int(config.train.seed)
    seed_all(seed_base + max(rank, 0))

    run_dir = resolve_run_dir(args, config_name, rank)
    ckpt_dir = os.path.join(run_dir, "ckpt_main")

    if is_main_process(rank):
        os.makedirs(ckpt_dir, exist_ok=True)
        snapshot_configs(run_dir, args, config=config)
    if is_ddp:
        dist.barrier()

    # -----------------------------------------------------
    # Logging / writer
    # -----------------------------------------------------
    logger, file_logger = setup_logger(run_dir, rank, debug=args.debug)
    info_logger = logger if args.debug else file_logger

    writer = None
    if is_main_process(rank) and not args.debug:
        writer = torch.utils.tensorboard.SummaryWriter(os.path.join(run_dir, "tb_main"))

    info_logger.info(args)
    info_logger.info(config)

    # -----------------------------------------------------
    # Datasets / samplers / loaders
    # -----------------------------------------------------
    info_logger.info("Loading dataset...")
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)

    train_sampler = None
    val_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    train_iterator = infinite_loader(train_loader, sampler=train_sampler, start_epoch=0)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    info_logger.info(f"Train {len(train_dataset)} | Val {len(val_dataset)}")

    # schedule = resolve_training_schedule(
    #     train_dataset_size=len(train_dataset),
    #     batch_size_per_gpu=config.train.batch_size,
    #     accumulation_steps=args.accumulation_steps,
    #     world_size=world_size,
    #     train_epochs=config.train.train_epochs,
    # )

    # max_global_steps = int(schedule["max_global_steps"])

    # info_logger.info(
    #     "[Schedule] train_size=%d | world_size=%d | batch_size_per_gpu=%d | accum=%d | "
    #     "effective_global_batch=%d | updates_per_epoch=%d | train_epochs=%d | max_global_steps=%d"
    #     % (
    #         schedule["train_dataset_size"],
    #         schedule["world_size"],
    #         schedule["batch_size_per_gpu"],
    #         schedule["accumulation_steps"],
    #         schedule["effective_global_batch"],
    #         schedule["updates_per_epoch"],
    #         schedule["train_epochs"],
    #         schedule["max_global_steps"],
    #     )
    # )
        # -----------------------------------------------------
    # Schedule: ABX-style conversion from single-GPU reference iters
    # -----------------------------------------------------
    schedule = resolve_max_global_steps(
        target_total_samples=int(config.train.target_total_samples),
        batch_size_per_gpu=config.train.batch_size,
        accumulation_steps=args.accumulation_steps,
        world_size=world_size,
    )

    max_global_steps = int(schedule["max_global_steps"])

    info_logger.info(
        "[Schedule] target_total_samples=%d | world_size=%d | batch_size_per_gpu=%d | accum=%d | "
        "effective_global_batch=%d | max_global_steps=%d"
        % (
            schedule["target_total_samples"],
            schedule["world_size"],
            schedule["batch_size_per_gpu"],
            schedule["accumulation_steps"],
            schedule["effective_global_batch"],
            schedule["max_global_steps"],
        )
    )
    
    
    # -----------------------------------------------------
    # Model / optimizer / scheduler / EMA
    # -----------------------------------------------------
    info_logger.info("Building model...")
    model = get_model(config.model).to(args.device)
    info_logger.info("Number of parameters: %d" % count_parameters(model))

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    ema = None
    if args.use_ema:
        ema = EMA(unwrap_model(model), decay=args.ema_decay)
        info_logger.info(f"[EMA] enabled with decay={args.ema_decay}")

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    # -----------------------------------------------------
    # Checkpoint manager
    # -----------------------------------------------------
    cfg_best_keys = getattr(config.train, "save_best_keys", None)
    cli_best_keys = args.save_best_keys if args.save_best_keys is not None else None
    best_keys = cli_best_keys if cli_best_keys is not None else (cfg_best_keys or [])
    ckpt_mgr = CheckpointManager(ckpt_dir, best_keys)

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------
    it_first = 1
    global_step_first = 0

    if args.resume is not None:
        info_logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)

        it_first = int(ckpt.get("iteration", 0)) + 1
        global_step_first = int(ckpt.get("global_step", 0))

        model_state = ckpt.get("model_raw", None)
        if model_state is None:
            model_state = ckpt["model"]

        unwrap_model(model).load_state_dict(model_state, strict=True)

        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            logger.warning(f"Optimizer state not loaded: {e}")

        if ckpt.get("scheduler", None) is not None and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                logger.warning(f"Scheduler state not loaded: {e}")

        if ema is not None and ckpt.get("ema", None) is not None:
            try:
                ema.load_state_dict(ckpt["ema"])
                if ema.num_updates is None and ema.use_num_updates:
                    ema.num_updates = global_step_first
                info_logger.info("[EMA] state resumed.")
            except Exception as e:
                logger.warning(f"EMA state not loaded: {e}")

        if "best_metrics" in ckpt and isinstance(ckpt["best_metrics"], dict):
            for k, v in ckpt["best_metrics"].items():
                if k in ckpt_mgr.best_metrics:
                    ckpt_mgr.best_metrics[k] = float(v)

        ckpt_mgr.sync_from_disk()
        info_logger.info(f"Resumed at micro-step {it_first}, global_step {global_step_first}.")

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------
    pbar = tqdm(
        initial=global_step_first,
        total=max_global_steps,
        desc="Train",
        dynamic_ncols=True,
        disable=not is_main_process(rank),
    )

    it = it_first
    global_step = global_step_first
    optimizer.zero_grad(set_to_none=True)

    try:
        while global_step < max_global_steps:
            accum_loss_dict = {}

            ddp_no_sync_model = hasattr(model, "no_sync")

            for micro_step in range(args.accumulation_steps):
                sync_ctx = model.no_sync() if (ddp_no_sync_model and micro_step < args.accumulation_steps - 1) else nullcontext()

                with sync_ctx:
                    unwrap_model(model).train()

                    batch = recursive_to(next(train_iterator), args.device)

                    loss_dict = model(batch)
                    raw_loss = sum_weighted_losses(loss_dict, config.train.loss_weights)

                    if not torch.isfinite(raw_loss):
                        raise RuntimeError(f"NaN/Inf loss detected at micro-step {it}")

                    if "overall" not in loss_dict:
                        loss_dict["overall"] = raw_loss

                    for k, v in loss_dict.items():
                        accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + float(v.detach().item())

                    (raw_loss / args.accumulation_steps).backward()

                it += 1

            global_step += 1
            pbar.update(1)

            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            if getattr(config.train.scheduler, "type", "") != "plateau":
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update()

            avg_loss_dict = {k: v / args.accumulation_steps for k, v in accum_loss_dict.items()}
            avg_loss_dict = reduce_dict_mean(avg_loss_dict, args.device)

            grad_norm_float = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            grad_norm_float = reduce_scalar(grad_norm_float, args.device) / world_size

            if is_main_process(rank):
                if global_step % args.log_steps == 0:
                    log_losses(
                        avg_loss_dict,
                        global_step,
                        "train",
                        info_logger,
                        writer if writer is not None else BlackHole(),
                        others={
                            "grad": grad_norm_float,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                    )
                pbar.set_postfix(
                    loss=f"{avg_loss_dict.get('overall', 0.0):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
                and global_step % config.train.val_freq != 0
                and is_main_process(rank)
            ):
                ckpt_mgr.save(
                    tag="last",
                    metric_value=float(avg_loss_dict.get("overall", 0.0)),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    config=config,
                    args=args,
                    iteration=it - 1,
                    global_step=global_step,
                    extra_state={"avg_train_loss": avg_loss_dict},
                )

            if global_step > 0 and global_step % config.train.val_freq == 0:
                avg_val = validate(
                    global_step=global_step,
                    model=model,
                    ema=ema,
                    val_loader=val_loader,
                    config=config,
                    args=args,
                    writer=writer,
                    rank=rank,
                    device=args.device,
                )

                if getattr(config.train.scheduler, "type", "") == "plateau":
                    scheduler.step(float(avg_val["overall"]))

                if is_main_process(rank):
                    ckpt_mgr.save(
                        tag="last",
                        metric_value=float(avg_val["overall"]),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        config=config,
                        args=args,
                        iteration=it - 1,
                        global_step=global_step,
                        extra_state={"avg_val_loss": avg_val},
                    )

                    if float(avg_val["overall"]) < ckpt_mgr.best_metrics["overall"]:
                        path = ckpt_mgr.save(
                            tag="best_overall",
                            metric_value=float(avg_val["overall"]),
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ema=ema,
                            config=config,
                            args=args,
                            iteration=it - 1,
                            global_step=global_step,
                            extra_state={"avg_val_loss": avg_val},
                        )
                        info_logger.info(
                            f"[Best overall] step={global_step} overall={avg_val['overall']:.4f} "
                            f"-> {os.path.basename(path)}"
                        )

                    for k in ckpt_mgr.best_keys:
                        if k == "overall":
                            continue
                        if k not in avg_val:
                            continue
                        if float(avg_val[k]) < ckpt_mgr.best_metrics[k]:
                            path = ckpt_mgr.save(
                                tag=f"best_{k}",
                                metric_value=float(avg_val[k]),
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                ema=ema,
                                config=config,
                                args=args,
                                iteration=it - 1,
                                global_step=global_step,
                                extra_state={"avg_val_loss": avg_val},
                            )
                            info_logger.info(
                                f"[Best {k}] step={global_step} {k}={avg_val[k]:.4f} "
                                f"-> {os.path.basename(path)}"
                            )

        if is_main_process(rank):
            info_logger.info("Training finished.")
        pbar.close()

    except KeyboardInterrupt:
        if is_main_process(rank):
            info_logger.info("Training interrupted by user.")
        pbar.close()
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        pbar.close()
        raise
    finally:
        if writer is not None:
            writer.close()
        ddp_cleanup()


if __name__ == "__main__":
    main()
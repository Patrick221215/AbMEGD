# inference.py
import os
import json
import time
import argparse
import logging
import multiprocessing as mp
from threading import Thread, Event
from logging.handlers import QueueHandler, QueueListener

import torch
from tqdm.auto import tqdm

from AbMEGD.datasets import get_dataset
from AbMEGD.utils.misc import load_config, seed_all

from AbMEGD.tools.runner.design_for_testset import (
    load_model_from_checkpoint,
    run_single_structure,
)


# =========================================================
# Logging: file-only, ABX-style
# =========================================================
def log_setup(run_dir, verbose=False):
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "inference.log")
    handlers = [logging.FileHandler(log_path, encoding="utf-8")]

    def handler_apply(h, f, *arg):
        f(*arg)
        return h

    level = logging.DEBUG if verbose else logging.INFO
    handlers = [handler_apply(h, h.setLevel, level) for h in handlers]

    fmt = "%(asctime)-15s [%(levelname)s] (%(process)d-%(filename)s:%(lineno)d) %(message)s"
    handlers = [handler_apply(h, h.setFormatter, logging.Formatter(fmt)) for h in handlers]

    logging.basicConfig(format=fmt, level=level, handlers=handlers, force=True)

    log_queue = mp.Queue(-1)
    return log_queue, handlers


def worker_setup(rank, log_queue, verbose=False):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


# =========================================================
# Utilities
# =========================================================
def resolve_runtime_seed(config, runtime_seed):
    if runtime_seed is not None:
        return int(runtime_seed)
    if hasattr(config, "sampling") and hasattr(config.sampling, "seed"):
        return int(config.sampling.seed)
    return 2024


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    return obj


def save_snapshot(run_dir, args, config):
    snap_dir = os.path.join(run_dir, "config_snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f_in:
            with open(os.path.join(snap_dir, os.path.basename(args.config)), "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())

    with open(os.path.join(run_dir, "runtime_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    try:
        with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
            json.dump(_to_plain_dict(config), f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def progress_monitor(done_counter, total_structures, stop_event):
    """
    Global run-level progress bar.
    Used mainly in multi-GPU mode, where worker-local tqdm would clash.
    """
    pbar = tqdm(total=total_structures, desc="Inference", dynamic_ncols=True, leave=True)
    last = 0
    try:
        while True:
            current = done_counter.value
            if current > last:
                pbar.update(current - last)
                last = current

            if stop_event.is_set() and current >= total_structures:
                break

            stop_event.wait(0.2)
    finally:
        # final sync
        current = done_counter.value
        if current > last:
            pbar.update(current - last)
        pbar.close()


def worker_inference(rank, log_queue, args, selected_indices, done_counter):
    worker_setup(rank, log_queue, args.verbose)
    logger = logging.getLogger(__name__)

    # Device
    if args.device == "gpu":
        gpu_index = args.gpu_list[rank]
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")

    # Config / seed / dataset / model
    config, _ = load_config(args.config)
    seed_all(resolve_runtime_seed(config, args.seed) + rank)

    dataset = get_dataset(config.dataset.test)
    model, ckpt, cfg_ckpt, lsd = load_model_from_checkpoint(
        model_path=args.model,
        device=device,
        use_ema=args.use_ema,
    )

    logger.info(
        f"[Worker {rank}] device={device} | assigned_structures={len(selected_indices)} | "
        f"use_ema={args.use_ema} | batch_size={args.batch_size} | num_samples={args.num_samples}"
    )
    logger.info(f"[Worker {rank}] load_state_dict result: {lsd}")

    structures_root = os.path.join(args.run_dir, "structures")
    os.makedirs(structures_root, exist_ok=True)

    # 单卡时显示每个结构内部的详细进度条；多卡时只显示主进程全局进度条
    show_local_pbar = (len(args.gpu_list) == 1)

    total_saved = 0
    for idx in selected_indices:
        structure = dataset[idx]
        structure_id = structure["id"]
        structure_out_dir = os.path.join(structures_root, f"{idx:04d}_{structure_id}")

        saved = run_single_structure(
            structure=structure,
            structure_index=idx,
            config=config,
            model=model,
            device=device,
            structure_out_dir=structure_out_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
            show_pbar=show_local_pbar,
        )
        total_saved += saved

        with done_counter.get_lock():
            done_counter.value += 1

    logger.info(f"[Worker {rank}] finished | total_saved={total_saved}")


def main(args):
    config, config_name = load_config(args.config)

    # Build run_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag_postfix = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, f"{config_name}{tag_postfix}", timestamp)
    args.run_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    mp.set_start_method("spawn", force=True)
    log_queue, handlers = log_setup(run_dir, args.verbose)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Snapshot
    save_snapshot(run_dir, args, config)

    # Dataset size / selected indices
    dataset = get_dataset(config.dataset.test)
    total_n = len(dataset)

    start_idx = max(0, int(args.start_idx))
    end_idx = total_n - 1 if int(args.end_idx) < 0 else min(int(args.end_idx), total_n - 1)

    if start_idx > end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}, dataset_size={total_n}")

    selected_indices = list(range(start_idx, end_idx + 1))

    manifest = {
        "config": args.config,
        "model": args.model,
        "use_ema": bool(args.use_ema),
        "output_dir": args.output_dir,
        "run_dir": run_dir,
        "mode": config.mode,
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "dataset_size": total_n,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_selected_structures": len(selected_indices),
        "gpu_list": args.gpu_list,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info("--------------------------------------------------")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Config       : {args.config}")
    logging.info(f"Model        : {args.model}")
    logging.info(f"use_ema      : {args.use_ema}")
    logging.info(f"Dataset size : {total_n}")
    logging.info(f"Selected     : {start_idx}..{end_idx} ({len(selected_indices)} structures)")
    logging.info(f"GPUs         : {args.gpu_list} | device={args.device}")
    logging.info("--------------------------------------------------")

    # Progress monitor
    done_counter = mp.Value("i", 0)
    stop_event = Event()
    monitor_thread = None

    # 多卡：主进程显示全局结构级总进度条
    if len(args.gpu_list) > 1 and args.device == "gpu":
        monitor_thread = Thread(
            target=progress_monitor,
            args=(done_counter, len(selected_indices), stop_event),
            daemon=True,
        )
        monitor_thread.start()

        sharded_indices = [selected_indices[r::len(args.gpu_list)] for r in range(len(args.gpu_list))]
        mp.spawn(
            worker_inference,
            args=(log_queue, args, sharded_indices, done_counter),
            nprocs=len(args.gpu_list),
            join=True,
        )
        stop_event.set()
        monitor_thread.join()
    else:
        # 单卡：详细内部进度条由 design_testset.run_single_structure 提供
        worker_inference(0, log_queue, args, selected_indices, done_counter)

    listener.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_list", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")

    parser.add_argument("--config", type=str, required=True, help="Path to AbMEGD inference task YAML")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output dir")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size PER GPU")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
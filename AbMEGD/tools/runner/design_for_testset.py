# AbMEGD/tools/runner/design_testset.py
import os
import argparse
import copy
import json
import logging

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from AbMEGD.datasets import get_dataset
from AbMEGD.models import get_model
from AbMEGD.modules.common.geometry import reconstruct_backbone_partially
from AbMEGD.modules.common.so3 import so3vec_to_rotation
from AbMEGD.utils.inference import RemoveNative
from AbMEGD.utils.protein.writers import save_pdb
from AbMEGD.utils.train import recursive_to
from AbMEGD.utils.misc import *
from AbMEGD.utils.data import *
from AbMEGD.utils.transforms import *
from AbMEGD.utils.inference import *


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure["id"]

    data_variants = []
    if config.mode == "single_cdr":
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                "data": data_var,
                "name": f"{structure_id}-{cdr_name}",
                "tag": f"{cdr_name}",
                "cdr": cdr_name,
                "residue_first": residue_first,
                "residue_last": residue_last,
            })

    elif config.mode == "multiple_cdrs":
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            "data": data_var,
            "name": f"{structure_id}-MultipleCDRs",
            "tag": "MultipleCDRs",
            "cdrs": cdrs,
            "residue_first": None,
            "residue_last": None,
        })

    elif config.mode == "full":
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            "data": data_var,
            "name": f"{structure_id}-Full",
            "tag": "Full",
            "residue_first": None,
            "residue_last": None,
        })

    elif config.mode == "abopt":
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    "data": data_var,
                    "name": f"{structure_id}-{cdr_name}-O{opt_step}",
                    "tag": f"{cdr_name}-O{opt_step}",
                    "cdr": cdr_name,
                    "opt_step": opt_step,
                    "residue_first": residue_first,
                    "residue_last": residue_last,
                })
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    return data_variants


def resolve_runtime_seed(config, runtime_seed):
    if runtime_seed is not None:
        return int(runtime_seed)
    if hasattr(config, "sampling") and hasattr(config.sampling, "seed"):
        return int(config.sampling.seed)
    return 2024


def resolve_runtime_model_path(config, runtime_model):
    if runtime_model is not None:
        return runtime_model
    if hasattr(config, "model") and hasattr(config.model, "checkpoint"):
        return config.model.checkpoint
    raise ValueError("No checkpoint specified. Please provide --model or put model.checkpoint in config.")


def resolve_runtime_num_samples(config, runtime_num_samples):
    if runtime_num_samples is not None:
        return int(runtime_num_samples)
    if hasattr(config, "sampling") and hasattr(config.sampling, "num_samples"):
        return int(config.sampling.num_samples)
    return 10


def load_model_from_checkpoint(model_path, device="cuda", use_ema=False):
    ckpt = torch.load(model_path, map_location="cpu")
    cfg_ckpt = ckpt["config"]

    model = get_model(cfg_ckpt.model).to(device)

    if use_ema:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt.get("model_raw", ckpt["model"])

    lsd = model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, ckpt, cfg_ckpt, lsd


@torch.no_grad()
def run_single_structure(
    structure,
    structure_index,
    config,
    model,
    device,
    structure_out_dir,
    num_samples,
    batch_size,
    num_workers=0,
    logger=None,
    show_pbar=True,
):
    """
    Reusable core function for both:
      - design_testset.py (single-index rerun)
      - inference.py      (batch inference over many indices)

    Progress behavior:
      - show_pbar=True  : outer structure-level variant tqdm + inner batch tqdm
      - show_pbar=False : silent execution, suitable for multi-process workers
    """
    os.makedirs(structure_out_dir, exist_ok=True)
    logger = logger or logging.getLogger(__name__)

    structure_id = structure["id"]
    logger.info(f"Start structure index={structure_index}, id={structure_id}")

    # Reference/native
    data_native = MergeChains()(structure)
    save_pdb(data_native, os.path.join(structure_out_dir, "reference.pdb"))

    # Variants
    get_structure = lambda: copy.deepcopy(structure)
    data_variants = create_data_variants(
        config=config,
        structure_factory=get_structure,
    )

    metadata = {
        "identifier": structure_id,
        "index": structure_index,
        "num_samples": int(num_samples),
        "mode": config.mode,
        "items": [{kk: vv for kk, vv in var.items() if kk != "data"} for var in data_variants],
    }
    with open(os.path.join(structure_out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [PatchAroundAnchor()]
    if "abopt" not in config.mode:
        inference_tfm.append(RemoveNative(
            remove_structure=config.sampling.sample_structure,
            remove_sequence=config.sampling.sample_sequence,
        ))
    inference_tfm = Compose(inference_tfm)

    total_saved = 0

    variant_iter = data_variants
    if show_pbar:
        variant_iter = tqdm(
            data_variants,
            desc=f"{structure_index:04d}_{structure_id}",
            dynamic_ncols=True,
            leave=False,
            position=0,
        )

    for variant in variant_iter:
        variant_dir = os.path.join(structure_out_dir, variant["tag"])
        os.makedirs(variant_dir, exist_ok=True)

        logger.info(f"Sampling variant={variant['tag']} for structure={structure_id}")
        save_pdb(data_native, os.path.join(variant_dir, "REF1.pdb"))

        data_cropped = inference_tfm(copy.deepcopy(variant["data"]))
        data_list_repeat = [copy.deepcopy(data_cropped) for _ in range(num_samples)]
        loader = DataLoader(
            data_list_repeat,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(str(device).startswith("cuda")),
        )

        count = 0
        iterator = tqdm(
            loader,
            desc=f"{variant['tag']}",
            dynamic_ncols=True,
            leave=False,
            disable=not show_pbar,
            position=1 if show_pbar else 0,
        )

        for batch in iterator:
            batch = recursive_to(batch, device)
            model.eval()

            if "abopt" in config.mode:
                traj_batch = model.optimize(
                    batch,
                    opt_step=variant["opt_step"],
                    optimize_opt={
                        "pbar": False,
                        "sample_structure": config.sampling.sample_structure,
                        "sample_sequence": config.sampling.sample_sequence,
                    },
                )
            else:
                traj_batch = model.sample(
                    batch,
                    sample_opt={
                        "pbar": False,
                        "sample_structure": config.sampling.sample_structure,
                        "sample_sequence": config.sampling.sample_sequence,
                    },
                )

            aa_new = traj_batch[0][2]
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx=batch["pos_heavyatom"],
                R_new=so3vec_to_rotation(traj_batch[0][0]),
                t_new=traj_batch[0][1],
                aa=aa_new,
                chain_nb=batch["chain_nb"],
                res_nb=batch["res_nb"],
                mask_atoms=batch["mask_heavyatom"],
                mask_recons=batch["generate_flag"],
            )

            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_tmpl = variant["data"]
                aa = apply_patch_to_tensor(data_tmpl["aa"], aa_new[i], data_cropped["patch_idx"])
                mask_ha = apply_patch_to_tensor(data_tmpl["mask_heavyatom"], mask_atom_new[i], data_cropped["patch_idx"])
                pos_ha = apply_patch_to_tensor(
                    data_tmpl["pos_heavyatom"],
                    pos_atom_new[i] + batch["origin"][i].view(1, 1, 3).cpu(),
                    data_cropped["patch_idx"],
                )

                save_path = os.path.join(variant_dir, f"{count:04d}.pdb")
                save_pdb(
                    {
                        "chain_nb": data_tmpl["chain_nb"],
                        "chain_id": data_tmpl["chain_id"],
                        "resseq": data_tmpl["resseq"],
                        "icode": data_tmpl["icode"],
                        "aa": aa,
                        "mask_heavyatom": mask_ha,
                        "pos_heavyatom": pos_ha,
                    },
                    path=save_path,
                )
                count += 1
                total_saved += 1

        logger.info(f"Finished variant={variant['tag']} | saved={count}")

    logger.info(f"Finished structure index={structure_index}, id={structure_id}, total_saved={total_saved}")
    return total_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    parser.add_argument("-c", "--config", type=str, default="./configs/test/codesign_single_inference.yml")
    parser.add_argument("-m", "--model", type=str, default=None, help="Runtime checkpoint override")
    parser.add_argument("-o", "--out_root", type=str, default="./results")
    parser.add_argument("-t", "--tag", type=str, default="")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-n", "--num_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true", default=False)
    args = parser.parse_args()

    config, config_name = load_config(args.config)
    seed_all(resolve_runtime_seed(config, args.seed))

    dataset = get_dataset(config.dataset.test)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} out of range [0, {len(dataset)-1}]")

    structure = dataset[args.index]
    structure_id = structure["id"]

    tag_postfix = f"_{args.tag}" if args.tag else ""
    run_root = get_new_log_dir(
        os.path.join(args.out_root, config_name + tag_postfix),
        prefix=f"{args.index:04d}_{structure_id}",
    )

    logger = get_logger("design_testset", run_root)
    logger.info(f"Single-index inference | index={args.index} | id={structure_id}")

    model_path = resolve_runtime_model_path(config, args.model)
    num_samples = resolve_runtime_num_samples(config, args.num_samples)

    logger.info(f"Loading checkpoint: {model_path}")
    logger.info(f"use_ema={args.use_ema} | num_samples={num_samples} | batch_size={args.batch_size}")

    model, ckpt, cfg_ckpt, lsd = load_model_from_checkpoint(
        model_path=model_path,
        device=args.device,
        use_ema=args.use_ema,
    )
    logger.info(str(lsd))

    run_single_structure(
        structure=structure,
        structure_index=args.index,
        config=config,
        model=model,
        device=args.device,
        structure_out_dir=run_root,
        num_samples=num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logger=logger,
        show_pbar=True,
    )


if __name__ == "__main__":
    main()
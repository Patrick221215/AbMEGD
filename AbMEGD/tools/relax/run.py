from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from typing import Callable, Iterable, List, Tuple

try:
    from .base import (
        RelaxTask,
        derive_stage_roots,
        prepare_relax_tasks_AbMEGD,
        required_stage_names_for_pipeline,
        summarize_tasks,
    )
except ImportError:
    from base import (
        RelaxTask,
        derive_stage_roots,
        prepare_relax_tasks_AbMEGD,
        required_stage_names_for_pipeline,
        summarize_tasks,
    )

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            yield from self.iterable


def _run_openmm_worker(
    args: Tuple[RelaxTask, str, int, int, float, bool, str, bool, bool, bool],
) -> RelaxTask:
    task, platform, cpu_threads, max_iterations, stiffness_kcal_per_A2, soft_fail, prepare_mode, fallback_to_full_fix, add_missing_residues, strict_AbMEGD_physics = args
    try:
        from .openmm_relaxer import run_openmm
    except ImportError:
        from openmm_relaxer import run_openmm
    return run_openmm(
        task,
        platform=platform,
        cpu_threads=cpu_threads,
        max_iterations=max_iterations,
        stiffness_kcal_per_A2=stiffness_kcal_per_A2,
        soft_fail=soft_fail,
        prepare_mode=prepare_mode,
        fallback_to_full_fix=fallback_to_full_fix,
        add_missing_residues=add_missing_residues,
        strict_AbMEGD_physics=strict_AbMEGD_physics,
    )


def _run_pyrosetta_worker(
    args: Tuple[RelaxTask, bool, str, int, str],
) -> RelaxTask:
    task, move_bb, subset, max_iter, scorefxn = args
    try:
        from .pyrosetta_relaxer import run_pyrosetta
    except ImportError:
        from pyrosetta_relaxer import run_pyrosetta
    return run_pyrosetta(task, move_bb=move_bb, subset=subset, max_iter=max_iter, scorefxn=scorefxn)


def _run_pyrosetta_fixbb_worker(
    args: Tuple[RelaxTask, str, int, str],
) -> RelaxTask:
    task, subset, max_iter, scorefxn = args
    try:
        from .pyrosetta_relaxer import run_pyrosetta_fixbb
    except ImportError:
        from pyrosetta_relaxer import run_pyrosetta_fixbb
    return run_pyrosetta_fixbb(task, subset=subset, max_iter=max_iter, scorefxn=scorefxn)


def _execute_stage(
    *,
    tasks: List[RelaxTask],
    worker: Callable,
    worker_args_builder: Callable[[RelaxTask], tuple],
    num_workers: int,
    desc: str,
    start_method: str,
) -> List[RelaxTask]:
    if not tasks:
        return []

    payloads = [worker_args_builder(task) for task in tasks]
    if num_workers <= 1:
        return [worker(payload) for payload in tqdm(payloads, total=len(payloads), desc=desc)]

    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(worker, payloads)
        return list(tqdm(iterator, total=len(payloads), desc=desc))


def _filter_alive(tasks: Iterable[RelaxTask]) -> List[RelaxTask]:
    return [t for t in tasks if t.status != "failed"]


def _apply_task_sharding(tasks: List[RelaxTask], num_shards: int, shard_id: int) -> List[RelaxTask]:
    if num_shards <= 1:
        return tasks
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards - 1}], got {shard_id}")
    tasks = sorted(tasks, key=lambda t: t.in_path)
    return [task for idx, task in enumerate(tasks) if idx % num_shards == shard_id]


def _resolve_final_tag(pipeline: str) -> str:
    if pipeline == "pyrosetta_fixbb":
        return "fixbb"
    if pipeline == "openmm_only":
        return "openmm"
    return "rosetta"


def _detect_auto_openmm_platform(num_openmm_workers: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "CPU"
    if num_openmm_workers > 1:
        return "CPU"
    return "CUDA"


def _sanitize_resources(args: argparse.Namespace) -> argparse.Namespace:
    total_cpu = os.cpu_count() or 1

    if args.strict_AbMEGD_physics:
        args.openmm_platform = "CUDA" if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() else "CPU"
        args.openmm_workers = args.openmm_workers 
        args.openmm_prepare_mode = "full"
        args.openmm_no_fallback_full_fix = True
        args.openmm_add_missing_residues = True
        args.openmm_max_iterations = 0
        args.openmm_stiffness = 10.0
        args.openmm_soft_fail = False
        args.pyrosetta_subset = "nbrs"
        args.pyrosetta_max_iter = 1000
        args.pyrosetta_scorefxn = "ref2015"

    if args.openmm_platform == "AUTO":
        args.openmm_platform = _detect_auto_openmm_platform(args.openmm_workers)

    if args.openmm_platform == "CUDA" and args.openmm_workers > 1:
        logging.warning(
            "openmm_platform=CUDA with openmm_workers>1 is unsafe on shared servers; forcing openmm_workers=1."
        )
        args.openmm_workers = 1

    if args.openmm_platform == "CPU":
        args.openmm_cpu_threads = max(1, min(args.openmm_cpu_threads, total_cpu))
        max_safe_openmm_workers = max(1, total_cpu // args.openmm_cpu_threads)
        if args.openmm_workers > max_safe_openmm_workers:
            logging.warning(
                "openmm_workers * openmm_cpu_threads exceeds available CPU cores; "
                f"forcing openmm_workers={max_safe_openmm_workers}."
            )
            args.openmm_workers = max_safe_openmm_workers

    args.pyrosetta_workers = max(1, min(args.pyrosetta_workers, total_cpu))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="AbMEGD relax refactor with ABX-style static scheduling and split stage roots.")
    parser.add_argument("--root", required=True, help="AbMEGD output root to scan.")
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Sibling prefix for stage roots. Default: same as --root, producing <prefix>_openmm, <prefix>_rosetta, <prefix>_fixbb only when needed.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["openmm_pyrosetta", "pyrosetta", "pyrosetta_fixbb", "openmm_only"],
        default="openmm_pyrosetta",
    )
    parser.add_argument("--strict_AbMEGD_physics", action="store_true")
    parser.add_argument("--openmm_platform", choices=["AUTO", "CPU", "CUDA"], default="AUTO")
    parser.add_argument("--openmm_workers", type=int, default=1)
    parser.add_argument("--openmm_cpu_threads", type=int, default=16)
    parser.add_argument("--openmm_max_iterations", type=int, default=100)
    parser.add_argument("--openmm_stiffness", type=float, default=10.0)
    parser.add_argument("--openmm_prepare_mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--openmm_no_fallback_full_fix", action="store_true")
    parser.add_argument("--openmm_add_missing_residues", action="store_true")
    parser.add_argument("--openmm_soft_fail", action="store_true")
    parser.add_argument("--pyrosetta_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--pyrosetta_subset", choices=["all", "target", "nbrs"], default="nbrs")
    parser.add_argument("--pyrosetta_max_iter", type=int, default=1000)
    parser.add_argument("--pyrosetta_scorefxn", type=str, default="ref2015")
    parser.add_argument("--start_method", choices=["spawn", "forkserver", "fork"], default="spawn")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    output_prefix = os.path.abspath((args.output_prefix or args.root).rstrip("/"))
    stage_roots = derive_stage_roots(output_prefix)
    active_stage_names = required_stage_names_for_pipeline(args.pipeline)
    for stage_name in active_stage_names:
        os.makedirs(stage_roots[stage_name], exist_ok=True)

    log_candidates = [stage_roots[s] for s in active_stage_names if s in stage_roots]
    log_dir = log_candidates[0] if log_candidates else os.path.dirname(output_prefix)
    log_path = os.path.join(log_dir, "relax_pipeline.log")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    args = _sanitize_resources(args)

    final_tag = _resolve_final_tag(args.pipeline)
    logging.info("Building one-shot task list...")
    tasks = prepare_relax_tasks_AbMEGD(
        args.root,
        output_prefix,
        copy_refs=True,
        skip_finished=True,
        final_tag=final_tag,
        active_stage_names=active_stage_names,
    )
    summary = summarize_tasks(tasks)
    logging.info(f"Task summary (before sharding): {summary}")
    if args.num_shards > 1:
        tasks = _apply_task_sharding(tasks, args.num_shards, args.shard_id)
        logging.info(
            "Applied deterministic task sharding: shard %d / %d -> %d tasks",
            args.shard_id, args.num_shards, len(tasks)
        )
    summary = summarize_tasks(tasks)
    logging.info(f"Task summary (this run): {summary}")

    stage_msgs = []
    if "openmm" in active_stage_names:
        stage_msgs.append(f"OpenMM outputs under {stage_roots['openmm']}")
    if "rosetta" in active_stage_names:
        stage_msgs.append(f"PyRosetta outputs under {stage_roots['rosetta']}")
    if "fixbb" in active_stage_names:
        stage_msgs.append(f"Fixbb outputs under {stage_roots['fixbb']}")
    logging.info("Stage-root policy: %s", " ; ".join(stage_msgs))

    logging.info(
        f"OpenMM policy: platform={args.openmm_platform}, workers={args.openmm_workers}, "
        f"cpu_threads={args.openmm_cpu_threads}, prepare_mode={args.openmm_prepare_mode}, "
        f"max_iterations={args.openmm_max_iterations}, strict_AbMEGD_physics={args.strict_AbMEGD_physics}"
    )
    if not tasks:
        logging.info("No tasks found.")
        return

    if args.pipeline in {"openmm_pyrosetta", "openmm_only"}:
        logging.info(
            f"Stage 1: OpenMM on {args.openmm_platform} "
            f"(workers={args.openmm_workers}, cpu_threads={args.openmm_cpu_threads}) -> {stage_roots['openmm']}"
        )
        tasks = _execute_stage(
            tasks=tasks,
            worker=_run_openmm_worker,
            worker_args_builder=lambda t: (
                t,
                args.openmm_platform,
                args.openmm_cpu_threads,
                args.openmm_max_iterations,
                args.openmm_stiffness,
                args.openmm_soft_fail,
                args.openmm_prepare_mode,
                not args.openmm_no_fallback_full_fix,
                args.openmm_add_missing_residues,
                args.strict_AbMEGD_physics,
            ),
            num_workers=args.openmm_workers,
            desc="OpenMM relax",
            start_method=args.start_method,
        )
        alive = _filter_alive(tasks)
        logging.info(f"OpenMM finished. alive={len(alive)} failed={len(tasks)-len(alive)}")
        tasks = alive

        if not tasks:
            logging.warning("All tasks failed during OpenMM stage.")
            return

        if args.pipeline == "openmm_only":
            n_success = sum(1 for t in tasks if t.status != "failed")
            logging.info(f"Done. openmm_only success={n_success}, openmm_root={stage_roots['openmm']}")
            return

    if args.pipeline == "openmm_pyrosetta":
        logging.info(f"Stage 2: PyRosetta (workers={args.pyrosetta_workers}) -> {stage_roots['rosetta']}")
        tasks = _execute_stage(
            tasks=tasks,
            worker=_run_pyrosetta_worker,
            worker_args_builder=lambda t: (
                t,
                True,
                args.pyrosetta_subset,
                args.pyrosetta_max_iter,
                args.pyrosetta_scorefxn,
            ),
            num_workers=args.pyrosetta_workers,
            desc="PyRosetta relax",
            start_method=args.start_method,
        )

    elif args.pipeline == "pyrosetta":
        logging.info(f"PyRosetta only (workers={args.pyrosetta_workers}) -> {stage_roots['rosetta']}")
        tasks = _execute_stage(
            tasks=tasks,
            worker=_run_pyrosetta_worker,
            worker_args_builder=lambda t: (
                t,
                True,
                args.pyrosetta_subset,
                args.pyrosetta_max_iter,
                args.pyrosetta_scorefxn,
            ),
            num_workers=args.pyrosetta_workers,
            desc="PyRosetta relax",
            start_method=args.start_method,
        )

    elif args.pipeline == "pyrosetta_fixbb":
        logging.info(f"PyRosetta fixbb only (workers={args.pyrosetta_workers}) -> {stage_roots['fixbb']}")
        tasks = _execute_stage(
            tasks=tasks,
            worker=_run_pyrosetta_fixbb_worker,
            worker_args_builder=lambda t: (
                t,
                args.pyrosetta_subset,
                args.pyrosetta_max_iter,
                args.pyrosetta_scorefxn,
            ),
            num_workers=args.pyrosetta_workers,
            desc="PyRosetta fixbb",
            start_method=args.start_method,
        )

    n_success = sum(1 for t in tasks if t.status != "failed")
    n_failed = len(tasks) - n_success
    final_msg = [f"success={n_success}", f"failed={n_failed}"]
    if "openmm" in active_stage_names:
        final_msg.append(f"openmm:{stage_roots['openmm']}")
    if "rosetta" in active_stage_names:
        final_msg.append(f"rosetta:{stage_roots['rosetta']}")
    if "fixbb" in active_stage_names:
        final_msg.append(f"fixbb:{stage_roots['fixbb']}")
    logging.info("Done. %s", " ; ".join(final_msg))


if __name__ == "__main__":
    main()

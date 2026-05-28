import argparse
import functools
import logging
import multiprocessing as mp
import os
import traceback
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

import sys

try:
    from .base import EvalTask, TaskScanner
    from .energy import eval_imp
    from .similarity import eval_region_similarity
except ImportError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from base import EvalTask, TaskScanner
    from energy import eval_imp
    from similarity import eval_region_similarity


DISPLAY_METRICS = ["RMSD_region", "AAR_region", "IMP"]


def worker(task: EvalTask, use_energy: bool):
    try:
        task = eval_region_similarity(task)
        if use_energy:
            task = eval_imp(task)
        return task.to_report_dict()
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Worker exception on {task.in_path}: {e}\n{tb}")
        return None


def _print_named_summary(df: pd.DataFrame, group_col: str, metrics: List[str], title: str):
    cols = [m for m in metrics if m in df.columns]
    if not cols or group_col not in df.columns:
        return
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    grouped = df.groupby(group_col, dropna=False)[cols].mean(numeric_only=True)
    print(grouped.to_string(float_format=lambda x: f"{x:.6f}"))


def _print_overall(df: pd.DataFrame, metrics: List[str]):
    cols = [m for m in metrics if m in df.columns]
    if not cols:
        return
    print("\n" + "=" * 72)
    print("Overall Average Metrics")
    print("=" * 72)
    print(df[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))


def main(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Building one-shot evaluation task list...")
    scanner = TaskScanner(root=args.root, mode=args.mode, postfix=args.postfix)
    tasks = scanner.scan()

    if not tasks:
        logging.warning("No valid evaluation tasks found.")
        return

    logging.info(f"Found {len(tasks)} evaluation tasks.")
    func = functools.partial(worker, use_energy=(not args.no_energy))

    if args.num_workers <= 1:
        results = [func(t) for t in tqdm(tasks, desc="Evaluating")]
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(pool.imap(func, tasks), total=len(tasks), desc="Evaluating"))

    results = [r for r in results if r is not None]
    if not results:
        logging.warning("All evaluation tasks failed.")
        return

    df = pd.DataFrame(results)
    metric_cols = [c for c in DISPLAY_METRICS if c in df.columns]

    _print_overall(df, metric_cols)
    _print_named_summary(df, "region", metric_cols, "Average Metrics by Designed Region")
    _print_named_summary(df, "structure", metric_cols, "Average Metrics by Structure")

    output_csv = os.path.join(args.root, args.output_csv)
    df.to_csv(output_csv, index=False, float_format="%.6f")
    logging.info(f"Saved full per-sample evaluation table to {output_csv}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="One-shot AbMEGD evaluation with region-level AAR/RMSD and IMP.")
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--mode", choices=["legacy", "AbMEGD"], default="AbMEGD")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--no_energy", action="store_true", help="Skip PyRosetta IMP calculation.")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    main(args)

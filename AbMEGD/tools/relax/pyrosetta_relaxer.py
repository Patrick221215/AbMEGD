from __future__ import annotations

import logging

try:
    from .base import RelaxTask
except ImportError:
    from base import RelaxTask

_PYROSETTA_INITIALIZED = False


def _ensure_pyrosetta_initialized() -> None:
    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return

    import pyrosetta

    pyrosetta.init(
        " ".join(
            [
                "-mute", "all",
                "-use_input_sc",
                "-ignore_unrecognized_res",
                "-ignore_zero_occupancy", "false",
                "-load_PDB_components", "false",
                "-relax:default_repeats", "2",
                "-no_fconfig",
            ]
        )
    )
    _PYROSETTA_INITIALIZED = True


def get_scorefxn(scorefxn_name: str):
    import pyrosetta

    corrections = {
        "beta_july15": False,
        "beta_nov16": False,
        "gen_potential": False,
        "restore_talaris_behavior": False,
    }
    if "beta_july15" in scorefxn_name or "beta_nov15" in scorefxn_name:
        corrections["beta_july15"] = True
    elif "beta_nov16" in scorefxn_name:
        corrections["beta_nov16"] = True
    elif "genpot" in scorefxn_name:
        corrections["gen_potential"] = True
        pyrosetta.rosetta.basic.options.set_boolean_option("corrections:beta_july15", True)
    elif "talaris" in scorefxn_name:
        corrections["restore_talaris_behavior"] = True

    for corr, value in corrections.items():
        pyrosetta.rosetta.basic.options.set_boolean_option(f"corrections:{corr}", value)
    return pyrosetta.create_score_function(scorefxn_name)


class RelaxRegion:
    def __init__(self, scorefxn: str = "ref2015", max_iter: int = 1000, subset: str = "nbrs", move_bb: bool = True):
        _ensure_pyrosetta_initialized()
        from pyrosetta.rosetta.protocols.relax import FastRelax

        assert subset in ("all", "target", "nbrs")
        self.scorefxn = get_scorefxn(scorefxn)
        self.fast_relax = FastRelax()
        self.fast_relax.set_scorefxn(self.scorefxn)
        self.fast_relax.max_iter(max_iter)
        self.subset = subset
        self.move_bb = move_bb

    def __call__(self, pdb_path: str, flexible_residue_first, flexible_residue_last):
        import pyrosetta
        from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
        from pyrosetta.rosetta.core.select import residue_selector as selections
        from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action

        pose = pyrosetta.pose_from_pdb(pdb_path)
        original_pose = pose.clone()

        tf = TaskFactory()
        tf.push_back(operation.InitializeFromCommandline())
        tf.push_back(operation.RestrictToRepacking())

        if flexible_residue_first is None or flexible_residue_last is None:
            raise ValueError("PyRosetta relax requires flexible_residue_first/last from metadata.json")

        if flexible_residue_first[-1] == " ":
            flexible_residue_first = flexible_residue_first[:-1]
        if flexible_residue_last[-1] == " ":
            flexible_residue_last = flexible_residue_last[:-1]

        gen_selector = selections.ResidueIndexSelector()
        gen_selector.set_index_range(
            pose.pdb_info().pdb2pose(*flexible_residue_first),
            pose.pdb_info().pdb2pose(*flexible_residue_last),
        )
        nbr_selector = selections.NeighborhoodResidueSelector()
        nbr_selector.set_focus_selector(gen_selector)
        nbr_selector.set_include_focus_in_subset(True)

        if self.subset == "all":
            subset_selector = None
        elif self.subset == "nbrs":
            subset_selector = nbr_selector
        else:
            subset_selector = gen_selector

        if subset_selector is not None:
            prevent_repacking_rlt = operation.PreventRepackingRLT()
            prevent_subset_repacking = operation.OperateOnResidueSubset(
                prevent_repacking_rlt,
                subset_selector,
                flip_subset=True,
            )
            tf.push_back(prevent_subset_repacking)

        mmf = MoveMapFactory()
        if self.move_bb:
            mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
        if subset_selector is not None:
            mmf.add_chi_action(move_map_action.mm_enable, subset_selector)
        else:
            mmf.add_chi_action(move_map_action.mm_enable, gen_selector)
        mm = mmf.create_movemap_from_pose(pose)

        self.fast_relax.set_movemap(mm)
        self.fast_relax.set_task_factory(tf)
        self.fast_relax.apply(pose)

        e_before = self.scorefxn(original_pose)
        e_after = self.scorefxn(pose)
        return pose, e_before, e_after


def run_pyrosetta(
    task: RelaxTask,
    *,
    move_bb: bool = True,
    subset: str = "nbrs",
    max_iter: int = 1000,
    scorefxn: str = "ref2015",
    output_tag: str = "rosetta",
) -> RelaxTask:
    if not task.can_proceed():
        return task

    final_path = task.rosetta_path if output_tag == "rosetta" else task.fixbb_path
    if task.stage_exists(final_path):
        task.mark_success(final_path)
        task.add_message("reuse existing PyRosetta stage")
        return task

    try:
        minimizer = RelaxRegion(scorefxn=scorefxn, max_iter=max_iter, subset=subset, move_bb=move_bb)
        pose_min, e_before, e_after = minimizer(
            pdb_path=task.current_path,
            flexible_residue_first=task.flexible_residue_first,
            flexible_residue_last=task.flexible_residue_last,
        )
        task.ensure_parent_dir(final_path)
        pose_min.dump_pdb(final_path)
        task.mark_success(final_path)
        task.add_message(f"PyRosetta ok: {e_before:.3f} -> {e_after:.3f}")
    except Exception as e:
        msg = f"PyRosetta failed on {task.current_path}: {type(e).__name__}: {e}"
        logging.error(msg)
        task.mark_failure(msg)
    return task


def run_pyrosetta_fixbb(task: RelaxTask, *, subset: str = "nbrs", max_iter: int = 1000, scorefxn: str = "ref2015") -> RelaxTask:
    return run_pyrosetta(
        task,
        move_bb=False,
        subset=subset,
        max_iter=max_iter,
        scorefxn=scorefxn,
        output_tag="fixbb",
    )

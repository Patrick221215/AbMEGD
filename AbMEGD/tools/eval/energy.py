import logging
import os
import re
from typing import List, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

try:
    from .base import EvalTask
except ImportError:
    from base import EvalTask


_PYROSETTA_READY = False


def _ensure_pyrosetta_ready():
    global _PYROSETTA_READY
    if _PYROSETTA_READY:
        return
    import contextlib
    import io

    import pyrosetta

    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        pyrosetta.init(
            "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            "-ignore_zero_occupancy false -load_PDB_components true "
            "-relax:default_repeats 2 -no_fconfig -mute all"
        )
    _PYROSETTA_READY = True


def pyrosetta_interface_energy(pdb_path: str, interface: str) -> float:
    _ensure_pyrosetta_ready()
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return float(pose.scores["dG_separated"])


def _chain_seq_from_model(model, chain_id: str) -> str:
    try:
        residues = list(model[chain_id].get_residues())
    except KeyError:
        return ""
    aa = []
    for r in residues:
        try:
            aa.append(seq1(r.get_resname()))
        except Exception:
            continue
    return "".join(aa)


def _autodetect_heavy_light(model):
    from AbMEGD.datasets.numbering import renumber_ab_seq

    chain_ids = [c.id for c in model.get_chains()]
    heavy_cands, light_cands = [], []
    for cid in chain_ids:
        seq = _chain_seq_from_model(model, cid)
        if not seq:
            continue
        try:
            h = renumber_ab_seq(seq, allow=["H"], scheme="imgt")
            if h.get("domain_numbering") is not None:
                heavy_cands.append((cid, len(seq)))
        except Exception:
            pass
        try:
            l = renumber_ab_seq(seq, allow=["K", "L"], scheme="imgt")
            if l.get("domain_numbering") is not None:
                light_cands.append((cid, len(seq)))
        except Exception:
            pass
    heavy = max(heavy_cands, key=lambda x: x[1])[0] if heavy_cands else None
    light = max(light_cands, key=lambda x: x[1])[0] if light_cands else None
    return heavy, light


def _infer_hl_from_name_or_path(pdb_file: str, model=None) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(pdb_file))[0].split("@")[0]
    for stage in ("_relaxed", "_rosetta", "_openmm", "_fixbb"):
        if base.endswith(stage):
            base = base[: -len(stage)]
            break
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]

    jobdir = os.path.basename(os.path.dirname(os.path.dirname(pdb_file)))
    jparts = jobdir.split("_")
    year_idx = None
    for i, p in enumerate(jparts):
        if re.fullmatch(r"\d{4}", p):
            year_idx = i
            break
    if year_idx is not None and year_idx >= 4:
        chain_tokens = jparts[2:year_idx]
        if len(chain_tokens) >= 2:
            return chain_tokens[0], chain_tokens[1]

    if model is not None:
        return _autodetect_heavy_light(model)

    return None, None


def interface_improvement(pred_file: str, ref_file: str) -> float:
    parser = PDBParser(QUIET=True)
    ref_model = parser.get_structure("ref", ref_file)[0]
    heavy_chain_id, light_chain_id = _infer_hl_from_name_or_path(ref_file, model=ref_model)
    if not heavy_chain_id or not light_chain_id:
        raise ValueError(f"Cannot infer heavy/light chains for {os.path.basename(ref_file)}")

    all_chain_ids = [c.id for c in ref_model.get_chains()]
    antigen_chain_ids = [c for c in all_chain_ids if c not in {heavy_chain_id, light_chain_id}]
    if not antigen_chain_ids:
        return np.nan

    interface = f"{heavy_chain_id}{light_chain_id}_{''.join(antigen_chain_ids)}"
    dG_gen = pyrosetta_interface_energy(pred_file, interface)
    dG_ref = pyrosetta_interface_energy(ref_file, interface)
    return 1.0 if dG_gen < dG_ref else 0.0


def eval_imp(task: EvalTask):
    try:
        task.scores["IMP"] = interface_improvement(task.in_path, task.ref_path)
    except Exception as e:
        logging.error(f"Failed to calculate IMP for {task.in_path}: {e}")
        task.scores["IMP"] = np.nan
    return task

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

ResiduePosition = Tuple[str, int, str]

_AbMEGD_JOB_RE = re.compile(r"^\d{4}_.+_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$")
_AbMEGD_REGION_RE = re.compile(r"^[HL]_CDR[123](?:-O\d+)?$")
_SAMPLE_PDB_RE = re.compile(r"^\d{4}\.pdb$")
_REF_PDB_RE = re.compile(r"^REF\d+\.pdb$", re.IGNORECASE)


def normalize_residue_position(value: Any) -> Optional[ResiduePosition]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            chain, resseq = value
            return str(chain), int(resseq), " "
        if len(value) >= 3:
            chain, resseq, icode = value[:3]
            icode = " " if icode in (None, "") else str(icode)
            return str(chain), int(resseq), icode

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        m = re.match(r"^([A-Za-z0-9]+?)(\d+)([A-Za-z]?)$", s)
        if m:
            chain, resseq, icode = m.groups()
            return chain, int(resseq), (icode or " ")

    raise ValueError(f"Unsupported residue position format: {value!r}")


def derive_stage_roots(output_prefix: str) -> Dict[str, str]:
    output_prefix = os.path.abspath(output_prefix.rstrip("/"))
    return {
        "openmm": f"{output_prefix}_openmm",
        "rosetta": f"{output_prefix}_rosetta",
        "fixbb": f"{output_prefix}_fixbb",
    }


def required_stage_names_for_pipeline(pipeline: str) -> List[str]:
    if pipeline == "openmm_only":
        return ["openmm"]
    if pipeline == "pyrosetta":
        return ["openmm", "rosetta"]
    if pipeline == "pyrosetta_fixbb":
        return ["openmm", "fixbb"]
    if pipeline == "openmm_pyrosetta":
        return ["openmm", "rosetta"]
    raise ValueError(f"Unsupported pipeline: {pipeline}")


@dataclass
class RelaxTask:
    in_path: str
    current_path: str
    relative_path: str
    name: str
    tag: str
    metadata_path: str
    stage_root_openmm: str
    stage_root_rosetta: str
    stage_root_fixbb: str
    flexible_residue_first: Optional[ResiduePosition] = None
    flexible_residue_last: Optional[ResiduePosition] = None
    status: str = "created"
    messages: List[str] = field(default_factory=list)

    @property
    def openmm_path(self) -> str:
        return os.path.join(self.stage_root_openmm, self.relative_path)

    @property
    def rosetta_path(self) -> str:
        return os.path.join(self.stage_root_rosetta, self.relative_path)

    @property
    def relaxed_path(self) -> str:
        return self.rosetta_path

    @property
    def fixbb_path(self) -> str:
        return os.path.join(self.stage_root_fixbb, self.relative_path)

    def ensure_parent_dir(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def can_proceed(self) -> bool:
        return self.status != "failed"

    def add_message(self, msg: str) -> None:
        self.messages.append(msg)

    def set_current_path(self, path: str) -> None:
        self.current_path = path

    def mark_success(self, final_path: str) -> None:
        self.status = "success"
        self.current_path = final_path

    def mark_failure(self, msg: Optional[str] = None) -> None:
        self.status = "failed"
        if msg:
            self.add_message(msg)

    def stage_exists(self, path: str) -> bool:
        return os.path.exists(path) and os.path.getsize(path) > 0


def _load_metadata(job_dir: str) -> Dict[str, Any]:
    metadata_path = os.path.join(job_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_region_metadata(job_dir: str, region_name: str) -> Optional[Dict[str, Any]]:
    try:
        metadata = _load_metadata(job_dir)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    for item in metadata.get("items", []):
        if item.get("tag") == region_name:
            return item
    return None


def _make_task(
    *,
    input_path: str,
    relative_path: str,
    region_name: str,
    metadata_path: str,
    metadata_item: Dict[str, Any],
    stage_roots: Dict[str, str],
) -> RelaxTask:
    first = normalize_residue_position(metadata_item.get("residue_first"))
    last = normalize_residue_position(metadata_item.get("residue_last"))
    return RelaxTask(
        in_path=input_path,
        current_path=input_path,
        relative_path=relative_path,
        name=os.path.splitext(os.path.basename(input_path))[0],
        tag=region_name,
        metadata_path=metadata_path,
        stage_root_openmm=stage_roots["openmm"],
        stage_root_rosetta=stage_roots["rosetta"],
        stage_root_fixbb=stage_roots["fixbb"],
        flexible_residue_first=first,
        flexible_residue_last=last,
    )


def _copy_if_exists(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def _copy_job_sidecars(job_path: str, input_dir: str, stage_roots: Dict[str, str], active_stage_names: Sequence[str]) -> None:
    job_rel = os.path.relpath(job_path, input_dir)
    for stage_name in active_stage_names:
        root = stage_roots[stage_name]
        _copy_if_exists(
            os.path.join(job_path, "metadata.json"),
            os.path.join(root, job_rel, "metadata.json"),
        )
        _copy_if_exists(
            os.path.join(job_path, "reference.pdb"),
            os.path.join(root, job_rel, "reference.pdb"),
        )


def prepare_relax_tasks_AbMEGD(
    input_dir: str,
    output_prefix: str,
    *,
    copy_refs: bool = True,
    skip_finished: bool = True,
    final_tag: str = "rosetta",
    active_stage_names: Optional[Sequence[str]] = None,
) -> List[RelaxTask]:
    """
    ABX-style one-shot task builder for AbMEGD outputs.

    Output policy in v7:
    - Only create the stage roots required by the selected pipeline.
    - OpenMM mirror tree lives under <output_prefix>_openmm/
    - PyRosetta mirror tree lives under <output_prefix>_rosetta/
    - Fixbb mirror tree lives under <output_prefix>_fixbb/ only when explicitly requested.
    - metadata.json and reference.pdb from each job directory are copied into each active stage root.
    - REF*.pdb from each region directory are also copied into each active stage root.
    - Original input tree is never modified.
    - Stage files keep the original basename, because the stage root already encodes the stage.
    """
    input_dir = os.path.abspath(input_dir)
    stage_roots = derive_stage_roots(output_prefix)
    active_stage_names = list(active_stage_names or stage_roots.keys())
    for stage_name in active_stage_names:
        os.makedirs(stage_roots[stage_name], exist_ok=True)

    tasks: List[RelaxTask] = []

    try:
        entries = sorted(os.listdir(input_dir))
    except FileNotFoundError:
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    for job_name in entries:
        job_path = os.path.join(input_dir, job_name)
        if not os.path.isdir(job_path):
            continue
        if not _AbMEGD_JOB_RE.match(job_name):
            continue

        _copy_job_sidecars(job_path, input_dir, stage_roots, active_stage_names)

        for region_name in sorted(os.listdir(job_path)):
            region_path = os.path.join(job_path, region_name)
            if not os.path.isdir(region_path):
                continue
            if not _AbMEGD_REGION_RE.match(region_name):
                continue

            metadata_item = _get_region_metadata(job_path, region_name)
            if metadata_item is None:
                continue
            metadata_path = os.path.join(job_path, "metadata.json")

            if copy_refs:
                for fn in sorted(os.listdir(region_path)):
                    if not _REF_PDB_RE.match(fn):
                        continue
                    src = os.path.join(region_path, fn)
                    rel = os.path.relpath(src, input_dir)
                    for stage_name in active_stage_names:
                        dst = os.path.join(stage_roots[stage_name], rel)
                        _copy_if_exists(src, dst)

            for fn in sorted(os.listdir(region_path)):
                if not _SAMPLE_PDB_RE.match(fn):
                    continue

                src = os.path.join(region_path, fn)
                rel = os.path.relpath(src, input_dir)
                candidate = _make_task(
                    input_path=src,
                    relative_path=rel,
                    region_name=region_name,
                    metadata_path=metadata_path,
                    metadata_item=metadata_item,
                    stage_roots=stage_roots,
                )

                if candidate.stage_exists(candidate.openmm_path):
                    candidate.set_current_path(candidate.openmm_path)

                final_path = {
                    "rosetta": candidate.rosetta_path,
                    "fixbb": candidate.fixbb_path,
                    "openmm": candidate.openmm_path,
                }[final_tag]

                if skip_finished and candidate.stage_exists(final_path):
                    continue
                tasks.append(candidate)

    return tasks


def summarize_tasks(tasks: Sequence[RelaxTask]) -> Dict[str, int]:
    num_with_metadata = sum(
        1 for t in tasks if t.flexible_residue_first is not None and t.flexible_residue_last is not None
    )
    num_reuse_openmm = sum(1 for t in tasks if t.current_path == t.openmm_path and t.stage_exists(t.openmm_path))
    return {
        "total": len(tasks),
        "with_flexible_region": num_with_metadata,
        "without_flexible_region": len(tasks) - num_with_metadata,
        "reuse_existing_openmm_inputs": num_reuse_openmm,
    }

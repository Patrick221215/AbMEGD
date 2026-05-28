import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Bio import PDB


@dataclass
class EvalTask:
    in_path: str
    ref_path: str
    info: Dict
    structure: str
    name: str
    method: str
    region: str
    ab_chains: List[str]
    sample_id: str = ""

    residue_first: Optional[Tuple] = None
    residue_last: Optional[Tuple] = None
    scores: Dict = field(default_factory=dict)

    def get_gen_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.in_path, self.in_path)[0]

    def get_ref_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.ref_path, self.ref_path)[0]

    def to_report_dict(self) -> Dict:
        return {
            "method": self.method,
            "structure": self.structure,
            "region": self.region,
            "sample_id": self.sample_id,
            "filename": os.path.basename(self.in_path),
            **self.scores,
        }


class TaskScanner:
    def __init__(self, root: str, mode: str = "AbMEGD", postfix: str = ""):
        self.root = os.path.abspath(root)
        self.mode = mode
        self.postfix = (postfix or "").strip()

    # ------------------------------------------------------------------
    # generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_stage_suffix(stem: str, postfix: str) -> str:
        if postfix and stem.endswith(f"_{postfix}"):
            return stem[: -(len(postfix) + 1)]
        return stem

    @staticmethod
    def _pick_best_regions(region_names: List[str]) -> List[str]:
        """
        If both H_CDR3 and H_CDR3-O10 exist, keep the largest optimization step.
        """
        best = {}
        for rn in region_names:
            if "-O" in rn:
                base, step_str = rn.split("-O", 1)
                try:
                    step = int(step_str)
                except ValueError:
                    step = 0
            else:
                base, step = rn, 0
            cur = best.get(base)
            if cur is None or step > cur[0]:
                best[base] = (step, rn)
        return [v[1] for v in best.values()]

    @staticmethod
    def _region_dir_re():
        return re.compile(r"^[HL]_CDR[123](?:-O\d+)?$")

    def _region_sample_predicate(self, fname: str) -> bool:
        """
        Accept:
          0000.pdb
          0000_rosetta.pdb   (when postfix=rosetta)
          0000_openmm.pdb    (when postfix=openmm)
        """
        if not fname.endswith(".pdb"):
            return False
        if fname.upper().startswith("REF"):
            return False

        stem = os.path.splitext(fname)[0]
        if self.postfix:
            return bool(re.fullmatch(rf"\d+(?:_{re.escape(self.postfix)})?", stem))
        return bool(re.fullmatch(r"\d+", stem))

    def _region_ref_candidates(self, region_path: str, structure_path: str) -> List[str]:
        cands = []
        if self.postfix:
            cands.extend(
                [
                    os.path.join(region_path, f"REF1_{self.postfix}.pdb"),
                    os.path.join(region_path, f"REF_{self.postfix}.pdb"),
                    os.path.join(structure_path, f"reference_{self.postfix}.pdb"),
                ]
            )
        cands.extend(
            [
                os.path.join(region_path, "REF1.pdb"),
                os.path.join(region_path, "REF.pdb"),
                os.path.join(structure_path, "reference.pdb"),
            ]
        )
        return cands

    def _get_metadata(self, structure_path: str, region_name: str) -> Optional[Dict]:
        json_path = os.path.join(structure_path, "metadata.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

        antibody_chains = set()
        info = None
        for item in metadata.get("items", []):
            if item.get("tag") == region_name:
                info = item
            residue_first = item.get("residue_first")
            if residue_first:
                antibody_chains.add(residue_first[0])

        if info is None:
            return None

        info = dict(info)
        info["antibody_chains"] = list(sorted(antibody_chains))
        info["structure"] = metadata.get("identifier", os.path.basename(structure_path))
        return info

    # ------------------------------------------------------------------
    # legacy mode
    # ------------------------------------------------------------------
    def _legacy_predicate(self, fname: str) -> bool:
        if self.postfix:
            return bool(re.fullmatch(rf".+_{re.escape(self.postfix)}\.pdb", fname))
        return fname.endswith(".pdb") and not fname.startswith("REF")

    def _legacy_reference_candidates(self, pred_path: str) -> List[str]:
        reference_dir = os.path.join(self.root, "reference")
        fname = os.path.basename(pred_path)
        stem = os.path.splitext(fname)[0]
        stem = self._strip_stage_suffix(stem, self.postfix)
        candidates = [os.path.join(reference_dir, f"{stem}.pdb")]
        if self.postfix:
            candidates.append(os.path.join(reference_dir, f"{stem}_{self.postfix}.pdb"))
        return candidates

    def _scan_legacy(self) -> List[EvalTask]:
        tasks: List[EvalTask] = []
        reference_dir = os.path.join(self.root, "reference")
        root_stage = os.path.basename(self.root)
        stage_implied = ""
        for stage in ("rosetta", "openmm", "fixbb"):
            if root_stage.endswith(f"_{stage}"):
                stage_implied = stage
                break

        for parent, _, files in os.walk(self.root):
            if os.path.abspath(parent).startswith(os.path.abspath(reference_dir)):
                continue
            for fname in sorted(files):
                fpath = os.path.join(parent, fname)
                if not self._legacy_predicate(fname):
                    if not (stage_implied and fname.endswith(".pdb") and not fname.startswith("REF")):
                        continue
                if os.path.getsize(fpath) == 0:
                    continue

                ref_path = None
                for cand in self._legacy_reference_candidates(fpath):
                    if os.path.exists(cand):
                        ref_path = cand
                        break
                if ref_path is None:
                    continue

                sample_id = os.path.splitext(fname)[0]
                region = "legacy"
                tasks.append(
                    EvalTask(
                        in_path=fpath,
                        ref_path=ref_path,
                        info={},
                        structure=sample_id,
                        name=sample_id,
                        method=os.path.basename(self.root),
                        region=region,
                        ab_chains=[],
                        sample_id=sample_id,
                    )
                )
        return tasks

    # ------------------------------------------------------------------
    # AbMEGD/new-inference mode
    # ------------------------------------------------------------------
    def _is_old_AbMEGD_jobdir(self, dirname: str) -> bool:
        job_dir_re = re.compile(
            r"^(?P<idx>\d{4})_(?P<pdbid>[^_]+)_(?P<chains>.+?)_"
            r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})__"
            r"(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})$"
        )
        return job_dir_re.match(dirname) is not None

    def _is_new_inference_structure_dir(self, dirpath: str) -> bool:
        """
        New inference layout:
          structures/
            0000_xxx/
              metadata.json
              reference.pdb
              H_CDR1/
              ...
        """
        if not os.path.isdir(dirpath):
            return False
        if not os.path.exists(os.path.join(dirpath, "metadata.json")):
            return False
        region_dir_re = self._region_dir_re()
        region_dirs = [
            rn for rn in os.listdir(dirpath)
            if os.path.isdir(os.path.join(dirpath, rn)) and region_dir_re.fullmatch(rn)
        ]
        return len(region_dirs) > 0

    def _collect_tasks_from_structure_dir(self, structure_path: str) -> List[EvalTask]:
        tasks: List[EvalTask] = []
        structure_name = os.path.basename(structure_path)

        region_dir_re = self._region_dir_re()
        region_names = [
            rn
            for rn in os.listdir(structure_path)
            if os.path.isdir(os.path.join(structure_path, rn)) and region_dir_re.fullmatch(rn)
        ]
        region_names = self._pick_best_regions(sorted(region_names))

        for region_name in region_names:
            region_path = os.path.join(structure_path, region_name)
            info = self._get_metadata(structure_path, region_name)
            if info is None:
                continue

            ref_path = None
            for cand in self._region_ref_candidates(region_path, structure_path):
                if os.path.exists(cand):
                    ref_path = cand
                    break
            if ref_path is None:
                continue

            for fname in sorted(os.listdir(region_path)):
                if not self._region_sample_predicate(fname):
                    continue

                pred_path = os.path.join(region_path, fname)
                if os.path.getsize(pred_path) == 0:
                    continue

                stem = os.path.splitext(fname)[0]
                if self.postfix and stem.endswith(f"_{self.postfix}"):
                    sample_id = stem[: -(len(self.postfix) + 1)]
                else:
                    sample_id = stem

                tasks.append(
                    EvalTask(
                        in_path=pred_path,
                        ref_path=ref_path,
                        info=info,
                        structure=info["structure"],
                        name=info.get("name", structure_name),
                        method=os.path.basename(self.root),
                        region=region_name,
                        ab_chains=info.get("antibody_chains", []),
                        sample_id=sample_id,
                        residue_first=info.get("residue_first"),
                        residue_last=info.get("residue_last"),
                    )
                )
        return tasks

    def _scan_AbMEGD(self) -> List[EvalTask]:
        tasks: List[EvalTask] = []

        # support both:
        # 1) old AbMEGD output root with timestamped jobdirs
        # 2) new inference output root = .../structures with 0000_xxx dirs
        for entry in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, entry)
            if not os.path.isdir(path):
                continue

            if self._is_old_AbMEGD_jobdir(entry):
                tasks.extend(self._collect_tasks_from_structure_dir(path))
            elif self._is_new_inference_structure_dir(path):
                tasks.extend(self._collect_tasks_from_structure_dir(path))

        return tasks

    # ------------------------------------------------------------------
    def scan(self) -> List[EvalTask]:
        if self.mode == "legacy":
            return self._scan_legacy()
        return self._scan_AbMEGD()
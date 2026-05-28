from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, Optional, Tuple

try:
    from .base import RelaxTask
except ImportError:
    from base import RelaxTask

ResiduePosition = Tuple[str, int, str]


_MINIMIZER_CACHE: Dict[Tuple[Any, ...], "ForceFieldMinimizer"] = {}


def _in_flexible_range(
    chain_id: str,
    res_id: int,
    ins_code: str,
    flexible_first: Optional[ResiduePosition],
    flexible_last: Optional[ResiduePosition],
) -> bool:
    if flexible_first is None or flexible_last is None:
        return False
    if chain_id != flexible_first[0]:
        return False
    return (flexible_first[1], flexible_first[2]) <= (res_id, ins_code) <= (flexible_last[1], flexible_last[2])


class ForceFieldMinimizer:
    """
    OpenMM minimizer that supports both:
    1. a strict AbMEGD-compatible preparation path; and
    2. lighter ABX-scheduler-friendly paths.
    """

    def __init__(
        self,
        *,
        stiffness_kcal_per_A2: float = 10.0,
        max_iterations: int = 100,
        platform: str = "CPU",
        platform_properties: Optional[Dict[str, str]] = None,
        forcefield_files: Tuple[str, ...] = ("amber99sb.xml",),
        use_hbonds_constraints: bool = True,
        tolerance_kj_per_nm: float = 10.0,
        prepare_mode: str = "fast",
        fallback_to_full_fix: bool = True,
        add_missing_residues: bool = False,
        strict_AbMEGD_physics: bool = False,
    ) -> None:
        if prepare_mode not in {"off", "fast", "full"}:
            raise ValueError(f"Unsupported prepare_mode: {prepare_mode}")
        self.stiffness_kcal_per_A2 = float(stiffness_kcal_per_A2)
        self.max_iterations = int(max_iterations)
        self.platform = platform
        self.platform_properties = platform_properties or ({"CudaPrecision": "mixed"} if platform == "CUDA" else {})
        self.forcefield_files = forcefield_files
        self.use_hbonds_constraints = bool(use_hbonds_constraints)
        self.tolerance_kj_per_nm = float(tolerance_kj_per_nm)
        self.prepare_mode = prepare_mode
        self.fallback_to_full_fix = bool(fallback_to_full_fix)
        self.add_missing_residues = bool(add_missing_residues)
        self.strict_AbMEGD_physics = bool(strict_AbMEGD_physics)
        self._cached_ff = None
        self._cached_platform = None

    @staticmethod
    def _lazy_imports():
        import pdbfixer
        import openmm
        from openmm import app as omm_app
        from openmm import unit as u
        return pdbfixer, openmm, omm_app, u

    def _get_forcefield(self):
        if self._cached_ff is None:
            _, _, omm_app, _ = self._lazy_imports()
            self._cached_ff = omm_app.ForceField(*self.forcefield_files)
        return self._cached_ff

    def _get_platform(self):
        if self._cached_platform is None:
            _, openmm, _, _ = self._lazy_imports()
            self._cached_platform = openmm.Platform.getPlatformByName(self.platform)
        return self._cached_platform

    def _prepare_pdb_fast(self, pdb_str: str) -> str:
        _, _, omm_app, _ = self._lazy_imports()
        ff = self._get_forcefield()
        pdb = omm_app.PDBFile(io.StringIO(pdb_str))
        modeller = omm_app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        buf = io.StringIO()
        omm_app.PDBFile.writeFile(modeller.topology, modeller.positions, buf, keepIds=True)
        return buf.getvalue()

    def _prepare_pdb_full(self, pdb_str: str) -> str:
        """
        Version-robust full pdbfixer path.

        The earlier bug came from skipping findMissingResidues() and then calling
        addMissingAtoms() on some pdbfixer versions where fixer.missingResidues is only
        created by findMissingResidues().

        To be robust across versions, we always call findMissingResidues(), and if the
        caller does NOT want missing residues added, we explicitly clear that dict.
        """
        pdbfixer, _, omm_app, _ = self._lazy_imports()
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        # Always initialize this attribute for compatibility with older/newer pdbfixer.
        fixer.findMissingResidues()
        if not self.add_missing_residues:
            fixer.missingResidues = {}

        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens()
        buf = io.StringIO()
        omm_app.PDBFile.writeFile(fixer.topology, fixer.positions, buf, keepIds=True)
        return buf.getvalue()

    def _prepare_pdb(self, pdb_str: str) -> str:
        if self.prepare_mode == "off":
            return pdb_str
        if self.prepare_mode == "full":
            return self._prepare_pdb_full(pdb_str)
        try:
            return self._prepare_pdb_fast(pdb_str)
        except Exception as e:
            if not self.fallback_to_full_fix:
                raise
            logging.warning(
                "Fast OpenMM preparation failed (%s: %s); falling back to full pdbfixer preparation.",
                type(e).__name__,
                e,
            )
            return self._prepare_pdb_full(pdb_str)

    def _pdb_to_string(self, topology, positions) -> str:
        _, _, omm_app, _ = self._lazy_imports()
        buf = io.StringIO()
        omm_app.PDBFile.writeFile(topology, positions, buf, keepIds=True)
        return buf.getvalue()

    def _minimize_once(
        self,
        pdb_str: str,
        flexible_residue_first: Optional[ResiduePosition],
        flexible_residue_last: Optional[ResiduePosition],
    ) -> Tuple[str, Dict[str, Any]]:
        _, openmm, omm_app, u = self._lazy_imports()
        pdb = omm_app.PDBFile(io.StringIO(pdb_str))

        constraints = omm_app.HBonds if self.use_hbonds_constraints else None
        ff = self._get_forcefield()
        system = ff.createSystem(pdb.topology, constraints=constraints)

        k_kj_per_nm2 = (
            self.stiffness_kcal_per_A2 * u.kilocalories_per_mole / (u.angstroms ** 2)
        ).value_in_unit(u.kilojoules_per_mole / (u.nanometers ** 2))

        posre = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        posre.addGlobalParameter("k", k_kj_per_nm2)
        for p in ("x0", "y0", "z0"):
            posre.addPerParticleParameter(p)

        positions_in_nm = pdb.positions.value_in_unit(u.nanometer)
        for i, atom in enumerate(pdb.topology.atoms()):
            chain_id = atom.residue.chain.id
            res_id = int(atom.residue.id)
            ins_code = atom.residue.insertionCode
            is_heavy = (atom.element is not None) and (atom.element.symbol != "H")
            if is_heavy and not _in_flexible_range(
                chain_id,
                res_id,
                ins_code,
                flexible_residue_first,
                flexible_residue_last,
            ):
                pos = positions_in_nm[i]
                posre.addParticle(i, [pos.x, pos.y, pos.z])

        system.addForce(posre)

        integrator = openmm.LangevinIntegrator(0 * u.kelvin, 1 / u.picosecond, 0.002 * u.picoseconds)
        platform = self._get_platform()
        sim = omm_app.Simulation(pdb.topology, system, integrator, platform, self.platform_properties)
        sim.context.setPositions(pdb.positions)

        ret: Dict[str, Any] = {}
        state0 = sim.context.getState(getEnergy=True, getPositions=True)
        ret["einit_kcal_per_mol"] = state0.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)

        tolerance = self.tolerance_kj_per_nm * u.kilojoules_per_mole / u.nanometer
        sim.minimizeEnergy(tolerance=tolerance, maxIterations=self.max_iterations)

        state1 = sim.context.getState(getEnergy=True, getPositions=True)
        ret["efinal_kcal_per_mol"] = state1.getPotentialEnergy().value_in_unit(u.kilocalories_per_mole)
        min_pdb = self._pdb_to_string(sim.topology, state1.getPositions())

        del state0, state1, sim, integrator, system, pdb
        return min_pdb, ret

    @staticmethod
    def _add_energy_remarks(pdb_str: str, ret: Dict[str, Any]) -> str:
        lines = pdb_str.splitlines()
        lines.insert(1, f"REMARK   1  FINAL ENERGY:   {ret['efinal_kcal_per_mol']:.3f} KCAL/MOL")
        lines.insert(1, f"REMARK   1  INITIAL ENERGY: {ret['einit_kcal_per_mol']:.3f} KCAL/MOL")
        return "\n".join(lines)

    def __call__(
        self,
        pdb_path: str,
        *,
        flexible_residue_first: Optional[ResiduePosition],
        flexible_residue_last: Optional[ResiduePosition],
    ) -> Tuple[str, Dict[str, Any]]:
        with open(pdb_path, "r", encoding="utf-8") as f:
            pdb_str = f.read()
        pdb_str = self._prepare_pdb(pdb_str)
        pdb_min, ret = self._minimize_once(
            pdb_str,
            flexible_residue_first=flexible_residue_first,
            flexible_residue_last=flexible_residue_last,
        )
        return self._add_energy_remarks(pdb_min, ret), ret


def run_openmm(
    task: RelaxTask,
    *,
    platform: str = "CPU",
    cpu_threads: int = 4,
    max_iterations: int = 100,
    stiffness_kcal_per_A2: float = 10.0,
    soft_fail: bool = True,
    prepare_mode: str = "fast",
    fallback_to_full_fix: bool = True,
    add_missing_residues: bool = False,
    strict_AbMEGD_physics: bool = False,
) -> RelaxTask:
    if not task.can_proceed():
        return task

    if task.stage_exists(task.openmm_path):
        task.set_current_path(task.openmm_path)
        task.add_message("reuse existing OpenMM stage")
        return task

    try:
        if platform == "CPU":
            os.environ["OPENMM_CPU_THREADS"] = str(cpu_threads)

        cache_key = (
            platform, cpu_threads, max_iterations, stiffness_kcal_per_A2,
            prepare_mode, fallback_to_full_fix, add_missing_residues, strict_AbMEGD_physics,
        )
        minimizer = _MINIMIZER_CACHE.get(cache_key)
        if minimizer is None:
            minimizer = ForceFieldMinimizer(
                platform=platform,
                max_iterations=max_iterations,
                stiffness_kcal_per_A2=stiffness_kcal_per_A2,
                prepare_mode=prepare_mode,
                fallback_to_full_fix=fallback_to_full_fix,
                add_missing_residues=add_missing_residues,
                strict_AbMEGD_physics=strict_AbMEGD_physics,
            )
            _MINIMIZER_CACHE[cache_key] = minimizer
        pdb_min, info = minimizer(
            task.current_path,
            flexible_residue_first=task.flexible_residue_first,
            flexible_residue_last=task.flexible_residue_last,
        )
        task.ensure_parent_dir(task.openmm_path)
        with open(task.openmm_path, "w", encoding="utf-8") as f:
            f.write(pdb_min)
        task.set_current_path(task.openmm_path)
        task.add_message(
            f"OpenMM ok: {info['einit_kcal_per_mol']:.3f} -> {info['efinal_kcal_per_mol']:.3f} kcal/mol"
        )
    except Exception as e:
        msg = f"OpenMM failed on {task.current_path}: {type(e).__name__}: {e}"
        logging.warning(msg)
        if soft_fail:
            task.add_message(msg)
            task.set_current_path(task.in_path)
        else:
            task.mark_failure(msg)
    return task

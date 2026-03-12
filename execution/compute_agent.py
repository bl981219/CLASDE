import logging
import os
import subprocess
import yaml
import json
import time
import hashlib
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

class SimulationType(str, Enum):
    DFT = "DFT"
    MLIP = "MLIP"
    MD = "MD"
    FINETUNING = "FINETUNING"

# --- Driver Architecture ---

class BaseComputeDriver(ABC):
    @abstractmethod
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        pass
    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        pass

class SlurmDriver(BaseComputeDriver):
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        slurm_cfg = profile.config.get("slurm", {})
        partition = resources.get("partition", slurm_cfg.get("partition", "normal"))
        ntasks = resources.get("ntasks", 48)
        nodes = resources.get("nodes", 2)
        run_cmd = profile.get_run_command(ntasks)
        
        script = f"""#!/bin/bash
#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --nodes={nodes}
#SBATCH --partition={partition}
{slurm_cfg.get('extra_header', '')}

{run_cmd}
"""
        with open(os.path.join(calc_dir, "submit.sh"), "w") as f: f.write(script)
        try:
            res = subprocess.run(["sbatch", "submit.sh"], cwd=calc_dir, capture_output=True, text=True)
            if res.returncode == 0: return res.stdout.strip().split()[-1]
        except: pass
        return "failed"

    def get_status(self, job_id: str) -> JobStatus:
        if job_id == "failed": return JobStatus.FAILED
        try:
            res = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
            if job_id not in res.stdout:
                return JobStatus.COMPLETED # Assumed if not in queue
            if " R " in res.stdout: return JobStatus.RUNNING
            if " PD " in res.stdout: return JobStatus.PENDING
        except: pass
        return JobStatus.RUNNING

class LocalDriver(BaseComputeDriver):
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        ntasks = resources.get("ntasks", 4)
        run_cmd = profile.get_run_command(ntasks)
        proc = subprocess.Popen(run_cmd, shell=True, cwd=calc_dir)
        return f"local_{proc.pid}"

    def get_status(self, job_id: str) -> JobStatus:
        pid = int(job_id.split("_")[1])
        try:
            os.kill(pid, 0)
            return JobStatus.RUNNING
        except OSError:
            return JobStatus.COMPLETED

class MockDriver(BaseComputeDriver):
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        ...
        with open(os.path.join(calc_dir, "results.json"), "w") as f: json.dump(results, f)
        return f"mock_{state_id[:8]}"

    def get_status(self, job_id: str) -> JobStatus:
        return JobStatus.COMPLETED


# --- Core Management ---

class ComputeProfile:
    def __init__(self, profile_path: Optional[str] = None):
        self.config = {"platform": "local", "run_command": "mpirun -np {ntasks} {executable}", 
                       "executable": os.getenv("VASP_EXECUTABLE", "vasp_std"),
                       "potcar_path": os.getenv("VASP_PP_PATH", ""), "slurm": {"partition": "xeon-p8", "extra_header": ""}}
        for path in [profile_path, "compute_profile.yaml", os.path.expanduser("~/.clasde/profile.yaml")]:
            if path and os.path.exists(path):
                with open(path, "r") as f: self.config.update(yaml.safe_load(f))
                break
    def get_run_command(self, ntasks: int) -> str:
        import re
        cmd = self.config["run_command"]
        cmd = re.sub(r"\{ntasks\}", str(ntasks), cmd)
        cmd = re.sub(r"\{executable\}", str(self.config["executable"]), cmd)
        return cmd

class ComputeManager:
    def __init__(self, config: Dict[str, Any]):
        from agents.builder_agent import StructureBuilder
        self.builder = StructureBuilder()
        self.profile = ComputeProfile(config.get("profile_path"))
        self.base_dir = "data/outputs"
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_jobs = {}
        
        # Priority: 1. Campaign Config, 2. Profile, 3. Default
        plat = config.get("platform") or self.profile.config.get("platform", "local")
        
        if plat == "hpc": 
            self.driver = SlurmDriver()
        elif plat == "mock": 
            self.driver = MockDriver()
        else: 
            self.driver = LocalDriver()
        
        logger.info(f"ComputeManager initialized with {plat} platform.")

    def check_readiness(self) -> bool:
        if self.profile.config["platform"] == "mock": return True
        exe = self.profile.config["executable"]
        if subprocess.run(["which", exe.split()[0]], capture_output=True).returncode != 0: return False
        p_path = self.profile.config["potcar_path"]
        return bool(p_path and os.path.exists(p_path))

    def submit_job(self, structure: Any, state: SurfaceState, sim_type: SimulationType = SimulationType.DFT, iteration: int = 0) -> str:
        state_id = state.get_id()
        calc_dir = os.path.join(self.base_dir, f"iter{iteration:03d}_{sim_type.value}_{state_id[:8]}")
        
        # Check if job already finished
        if os.path.exists(os.path.join(calc_dir, "results.json")):
            logger.info(f"Job in {calc_dir} already completed. Skipping submission.")
            job_id = f"prev_{state_id[:8]}"
            self.active_jobs[job_id] = {"dir": calc_dir, "status": JobStatus.COMPLETED}
            return job_id
            
        if os.path.exists(os.path.join(calc_dir, "OUTCAR")) and os.path.getsize(os.path.join(calc_dir, "OUTCAR")) > 10000:
            # Check if it's finished or crashed
            with open(os.path.join(calc_dir, "OUTCAR"), 'rb') as f:
                f.seek(-2000, os.SEEK_END) if os.path.getsize(os.path.join(calc_dir, "OUTCAR")) > 2000 else None
                last_lines = f.read().decode('utf-8', errors='ignore')
                if "General timing and accounting" in last_lines:
                    logger.info(f"VASP in {calc_dir} already finished. Skipping.")
                    job_id = f"prev_{state_id[:8]}"
                    self.active_jobs[job_id] = {"dir": calc_dir, "status": JobStatus.COMPLETED}
                    return job_id

        os.makedirs(calc_dir, exist_ok=True)
        
        if sim_type == SimulationType.DFT:
            # Check if there is an active Slurm job ID from a previous attempt
            # (We could search for slurm-*.out files)
            slurm_files = [f for f in os.listdir(calc_dir) if f.startswith("slurm-") and f.endswith(".out")]
            if slurm_files:
                # Get the latest one
                latest_slurm = sorted(slurm_files)[-1]
                job_id = latest_slurm.split("-")[1].split(".")[0]
                status = self.driver.get_status(job_id)
                if status in [JobStatus.RUNNING, JobStatus.PENDING]:
                    logger.info(f"Detected active Slurm job {job_id} for {calc_dir}. Re-attaching.")
                    self.active_jobs[job_id] = {"dir": calc_dir, "status": status}
                    return job_id

            # 1. Write POSCAR
            from ase.io import write
            write(os.path.join(calc_dir, "POSCAR"), structure, format="vasp")
            
            # 2. Write INCAR and KPOINTS
            self._write_vasp_incar(calc_dir)
            self._write_vasp_kpoints(calc_dir)
            
            # 3. Generate POTCAR from local library
            self._generate_potcar(calc_dir, structure)
            
            # 4. Dispatch via Driver
            job_id = self.driver.submit(calc_dir, state_id, self.profile.config.get("resources", {}), self.profile)
            self.active_jobs[job_id] = {"dir": calc_dir, "status": JobStatus.RUNNING}
            return job_id
        
        return self._handle_mlip_local(calc_dir, structure, state_id)

    def _write_vasp_incar(self, calc_dir: str) -> None:
        """Writes VASP INCAR with intelligent overrides and error recovery logic."""
        # Baseline Production Defaults
        params = {
            "PREC": "Accurate",
            "ENCUT": 450,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "NSW": 100,
            "IBRION": 2,
            "POTIM": 0.5,
            "LREAL": "Auto",
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "LORBIT": 11,
            "ALGO": "Fast"
        }
        
        # 1. Override with global compute profile settings
        params.update(self.profile.config.get("vasp_params", {}))
        
        # 2. Check for "Attempt Number" to apply recovery fixes
        # (This would be passed if we implemented a retry loop)
        
        with open(os.path.join(calc_dir, "INCAR"), "w") as f:
            for k, v in params.items():
                f.write(f"{k} = {v}\n")

    def _check_for_vasp_errors(self, calc_dir: str) -> Optional[str]:
        """Scans OUTCAR for known VASP crash patterns to trigger agent-led recovery."""
        outcar = os.path.join(calc_dir, "OUTCAR")
        if not os.path.exists(outcar): return None
        
        with open(outcar, "r") as f:
            content = f.read()
            if "ZPOTRF" in content: return "LAPACK_ERR"
            if "EDDDAV" in content: return "ELECTRONIC_DIVERGENCE"
            if "ZBRENT" in content: return "IONIC_DIVERGENCE"
        return None

    def _write_vasp_kpoints(self, calc_dir: str) -> None:
        kpoints_str = self.profile.config.get("kpoints", "Automatic\n0\nGamma\n1 1 1\n0 0 0\n")
        with open(os.path.join(calc_dir, "KPOINTS"), "w") as f:
            f.write(kpoints_str)

    def _generate_potcar(self, calc_dir: str, structure: Any) -> None:
        """Concatenates POTCAR files from the profile's potcar_path in the correct order."""
        pot_base = self.profile.config.get("potcar_path", "")
        if not pot_base or not os.path.exists(pot_base):
            logger.error(f"POTCAR path invalid: {pot_base}")
            return

        # VASP CRITICAL: POTCAR species must be unique and in the order of the POSCAR groups
        # ASE write_vasp groups atoms by species. We must match that.
        symbols = structure.get_chemical_symbols()
        unique_species = []
        for s in symbols:
            if s not in unique_species:
                unique_species.append(s)
            
        logger.info(f"Generating POTCAR for species: {unique_species}")
        with open(os.path.join(calc_dir, "POTCAR"), "wb") as out_f:
            for el in unique_species:
                # Search for el folder or el_pv/el_sv variants
                pot_file = None
                search_paths = [
                    os.path.join(pot_base, el, "POTCAR"),
                    os.path.join(pot_base, f"{el}_pv", "POTCAR"),
                    os.path.join(pot_base, f"{el}_sv", "POTCAR")
                ]
                for p in search_paths:
                    if os.path.exists(p):
                        pot_file = p
                        break
                
                if pot_file:
                    with open(pot_file, "rb") as in_f:
                        out_f.write(in_f.read())
                else:
                    logger.warning(f"POTCAR for {el} not found in {pot_base}")

    def _handle_mlip_local(self, calc_dir: str, structure: Any, state_id: str) -> str:
        try:
            from chgnet.model.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
            
            logger.info("Using CHGNet potential for local screening.")
            model = CHGNet.load()
            relaxer = StructOptimizer(model=model)
            
            # Configurable relaxation steps
            steps = self.profile.config.get("mlip_steps", 50)
            fmax = self.profile.config.get("mlip_fmax", 0.1)
            
            result = relaxer.relax(structure, fmax=fmax, steps=steps)
            res = {"total_energy": float(result["trajectory"].energies[-1]), "status": "completed", "fidelity": "MLIP (CHGNet)"}
        except Exception as e:
            logger.warning(f"CHGNet relaxation failed: {e}. Attempting stable EMT fallback.")
            try:
                from ase.calculators.emt import EMT
                from ase.optimize import BFGS
                # Note: EMT only supports late transition metals, but good for testing infrastructure
                structure.calc = EMT()
                dyn = BFGS(structure, logfile=None)
                dyn.run(fmax=0.2, steps=20)
                res = {"total_energy": float(structure.get_potential_energy()), "status": "completed", "fidelity": "MLIP (EMT-Fallback)"}
            except Exception as e2:
                logger.error(f"EMT Fallback also failed: {e2}")
                res = {"total_energy": 0.0, "status": "failed"}
        
        with open(os.path.join(calc_dir, "results.json"), "w") as f: json.dump(res, f)
        job_id = f"mlip_{state_id[:8]}"
        self.active_jobs[job_id] = {"dir": calc_dir, "status": JobStatus.COMPLETED}
        return job_id

    def fetch_results(self, job_id: str) -> str:
        return self.active_jobs[job_id]["dir"]

    def get_job_status(self, job_id: str) -> JobStatus:
        if job_id not in self.active_jobs:
            return JobStatus.FAILED
        
        status = self.driver.get_status(job_id)
        self.active_jobs[job_id]["status"] = status
        
        # Smell #3 Fix: Automated Recovery
        if status == JobStatus.FAILED:
            calc_dir = self.active_jobs[job_id]["dir"]
            error_type = self._check_for_vasp_errors(calc_dir)
            
            if error_type:
                logger.warning(f"Detected {error_type} in {calc_dir}. Attempting recovery...")
                return self.recover_job(job_id, error_type)
                
        return status

    def recover_job(self, job_id: str, error_type: str) -> JobStatus:
        """Applies physical heuristics to fix VASP crashes and resubmits."""
        calc_dir = self.active_jobs[job_id]["dir"]
        attempts = self.active_jobs[job_id].get("attempts", 1)
        
        if attempts >= 3:
            logger.error(f"Max recovery attempts reached for {calc_dir}. Marking as hard failure.")
            return JobStatus.FAILED

        # 1. Load existing parameters
        params = {}
        incar_path = os.path.join(calc_dir, "INCAR")
        with open(incar_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=")
                    params[k.strip()] = v.strip()

        # 2. Apply fixes based on error type
        if error_type == "LAPACK_ERR" or error_type == "ELECTRONIC_DIVERGENCE":
            # Slow down electronic convergence
            params["ALGO"] = "Normal"
            params["AMIX"] = "0.2"
            params["BMIX"] = "0.0001"
            logger.info("  -> Switching to ALGO=Normal and damping mixers.")
        elif error_type == "IONIC_DIVERGENCE":
            # Reduce step size
            params["POTIM"] = str(float(params.get("POTIM", 0.5)) / 2.0)
            logger.info(f"  -> Halving POTIM to {params['POTIM']}.")

        # 3. Rewrite INCAR
        with open(incar_path, "w") as f:
            for k, v in params.items(): f.write(f"{k} = {v}\n")

        # 4. Resubmit
        state_id = calc_dir.split("_")[-1]
        resources = self.profile.config.get("resources", {})
        new_job_id = self.driver.submit(calc_dir, state_id, resources, self.profile)
        
        # 5. Update tracking
        self.active_jobs[new_job_id] = {
            "dir": calc_dir, 
            "status": JobStatus.RUNNING,
            "attempts": attempts + 1
        }
        logger.info(f"  -> Resubmitted as job {new_job_id} (Attempt {attempts + 1})")
        return JobStatus.RUNNING

    def train_chgnet(self, db: Any):
        """Fine-tunes the CHGNet model using all results in the experiment DB."""
        dataset = db.get_training_data()
        if not dataset:
            logger.warning("No data found in ExperimentDB. Skipping fine-tuning.")
            return

        structures = []
        energies = []
        for entry in dataset:
            if entry.get("status") != "failed":
                state = entry["state"]
                # We need the physical Atoms object
                struct = entry["observables"].get("structure") or self.builder.build_structure(state)
                structures.append(struct)
                energies.append(entry["observables"].get("total_energy", 0.0))

        logger.info(f"--- Fine-tuning CHGNet on {len(structures)} structures ---")
        import sys
        sys.setrecursionlimit(2000)
        try:
            from chgnet.model.model import CHGNet
            from chgnet.trainer import Trainer
            from chgnet.data.dataset import StructureData, get_loader
            
            # Load base model
            model = CHGNet.load()
            
            # Prepare Data
            from pymatgen.io.ase import AseAtomsAdaptor
            pmg_structures = [AseAtomsAdaptor.get_structure(s) for s in structures]
            
            dataset = StructureData(
                structures=pmg_structures, 
                energies=energies, 
                stresses=None, 
                forces=None
            )
            train_loader, val_loader, _ = get_loader(dataset, batch_size=min(16, len(dataset)))
            
            trainer = Trainer(model=model)
            # Perform a rapid fine-tuning (e.g., 5 epochs for demo)
            trainer.train(train_loader, val_loader, epochs=5)
            model.save("finetuned_lscf_chgnet.pth")
            logger.info("CHGNet fine-tuning complete. Model saved to finetuned_lscf_chgnet.pth")
        except Exception as e:
            logger.error(f"CHGNet fine-tuning failed: {e}.")



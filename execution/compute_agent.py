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
    """Enumeration of possible HPC job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"

class SimulationType(str, Enum):
    """Categories of computational tasks."""
    DFT = "DFT"
    MLIP = "MLIP"
    MD = "MD"
    FINETUNING = "FINETUNING"

# --- Backend Architecture ---

class BaseComputeBackend(ABC):
    """
    Abstract base class for simulation engines. 
    Handles the 'How' of running a calculation.
    """
    @abstractmethod
    def setup(self, calc_dir: str, state: SurfaceState, profile: Any):
        """Prepare input files in the target directory."""
        pass

    @abstractmethod
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        """Dispatch the job and return a unique Job ID."""
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        """Query the status of a dispatched job."""
        pass

class VASPBackend(BaseComputeBackend):
    """Backend for VASP calculations on Slurm clusters."""
    
    def setup(self, calc_dir: str, state: SurfaceState, profile: Any):
        from ase.io import write
        ase_atoms = state.get_ase_atoms()
        if ase_atoms is None:
            raise ValueError("State contains no structural data for VASP setup.")
        
        # 1. POSCAR
        write(os.path.join(calc_dir, "POSCAR"), ase_atoms, format="vasp")
        
        # 2. INCAR (from profile defaults)
        params = profile.get("vasp_params", {})
        with open(os.path.join(calc_dir, "INCAR"), "w") as f:
            for k, v in params.items(): f.write(f"{k} = {v}\n")
            
        # 3. KPOINTS (from profile defaults)
        kpts = profile.get("kpoints")
        with open(os.path.join(calc_dir, "KPOINTS"), "w") as f: f.write(kpts)
        
        # 4. POTCAR (requires potcar_path in profile)
        self._generate_potcar(calc_dir, ase_atoms, profile.get("potcar_path", ""))

    def _generate_potcar(self, calc_dir: str, atoms: Any, pot_base: str):
        if not pot_base or not os.path.exists(pot_base):
            logger.error("Invalid POTCAR path.")
            return
        symbols = atoms.get_chemical_symbols()
        unique = []
        for s in symbols:
            if s not in unique: unique.append(s)
        
        with open(os.path.join(calc_dir, "POTCAR"), "wb") as out_f:
            for el in unique:
                for suffix in ["", "_pv", "_sv"]:
                    p = os.path.join(pot_base, f"{el}{suffix}", "POTCAR")
                    if os.path.exists(p):
                        with open(p, "rb") as in_f: out_f.write(in_f.read())
                        break

    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        partition = resources.get("partition", profile.get("slurm", {}).get("partition", "normal"))
        ntasks = resources.get("ntasks", 48)
        nodes = resources.get("nodes", 2)
        
        script = f"""#!/bin/bash
#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --nodes={nodes}
#SBATCH --partition={partition}
{profile.get("slurm", {}).get("extra_header", "")}

# Initialize environment for Supercloud
source /etc/profile

{profile.get_run_command(ntasks)}
"""
        with open(os.path.join(calc_dir, "submit.sh"), "w") as f: f.write(script)
        res = subprocess.run(["sbatch", "submit.sh"], cwd=calc_dir, capture_output=True, text=True)
        return res.stdout.strip().split()[-1] if res.returncode == 0 else "failed"

    def get_status(self, job_id: str) -> JobStatus:
        if job_id == "failed": return JobStatus.FAILED
        res = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in res.stdout: return JobStatus.COMPLETED
        return JobStatus.RUNNING if " R " in res.stdout else JobStatus.PENDING

class ASEBackend(BaseComputeBackend):
    """Local backend for relaxations using ASE potentials."""
    
    def setup(self, calc_dir: str, state: SurfaceState, profile: Any):
        pass

    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any], profile: Any) -> str:
        # In this simplistic version, we assume synchronous execution 
        # or that the manager handles the call to run_sync.
        return f"local_{state_id[:8]}"

    def run_sync(self, calc_dir: str, state: SurfaceState, profile: Any):
        try:
            atoms = state.get_ase_atoms()
            mode = profile.get("mode", "chgnet")
            
            if mode == "chgnet":
                from chgnet.model.model import CHGNet
                from chgnet.model.dynamics import StructOptimizer
                model = CHGNet.load()
                relaxer = StructOptimizer(model=model)
                result = relaxer.relax(atoms, fmax=0.1, steps=50)
                energy = float(result["trajectory"].energies[-1])
            else:
                from ase.calculators.emt import EMT
                from ase.optimize import BFGS
                atoms.calc = EMT()
                dyn = BFGS(atoms, logfile=None)
                dyn.run(fmax=0.2, steps=20)
                energy = float(atoms.get_potential_energy())
            
            res = {"total_energy": energy, "status": "completed", "fidelity": f"MLIP ({mode})"}
            with open(os.path.join(calc_dir, "results.json"), "w") as f: json.dump(res, f)
        except Exception as e:
            logger.error(f"ASE relaxation failed: {e}")
            with open(os.path.join(calc_dir, "results.json"), "w") as f:
                json.dump({"status": "failed", "error": str(e)}, f)

    def get_status(self, job_id: str) -> JobStatus:
        return JobStatus.COMPLETED

# --- Infrastructure ---

class ComputeProfile:
    def __init__(self, profile_path: Optional[str] = None):
        # Baseline Defaults (Academic Standard)
        self.config = {
            "platform": "local", 
            "run_command": "vasp_std", 
            "executable": "vasp_std",
            "vasp_params": {
                "PREC": "Accurate", "ENCUT": 450, "ISMEAR": 0, "SIGMA": 0.05,
                "NSW": 100, "IBRION": 2, "LORBIT": 11, "LREAL": "Auto"
            },
            "kpoints": "Automatic\n0\nGamma\n1 1 1\n0 0 0\n"
        }
        if profile_path and os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                self.config.update(yaml.safe_load(f))
    
    def get(self, key, default=None): return self.config.get(key, default)

    def get_run_command(self, ntasks: int) -> str:
        cmd = self.config.get("run_command", "{executable}")
        return cmd.format(ntasks=ntasks, executable=self.config.get("executable", "vasp_std"))

class ComputeManager:
    def __init__(self, config: Dict[str, Any]):
        from agents.builder_agent import StructureBuilder
        self.builder = StructureBuilder()
        self.profile = ComputeProfile(config.get("profile_path"))
        self.base_dir = "data/outputs"
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_jobs = {}
        
        plat = config.get("platform") or self.profile.get("platform", "local")
        self.backend = VASPBackend() if plat == "hpc" else ASEBackend()
        logger.info(f"ComputeManager using {self.backend.__class__.__name__}")

    def submit_job(self, structure: Any, state: SurfaceState, 
                   sim_type: SimulationType = SimulationType.DFT, 
                   iteration: int = 0) -> str:
        """Coordinates the setup and dispatch of a simulation task."""
        # Ensure sim_type is the correct Enum type
        if isinstance(sim_type, str):
            sim_type = SimulationType(sim_type.upper())

        state_id = state.get_id()
        calc_dir = os.path.join(self.base_dir, f"iter{iteration:03d}_{sim_type.value}_{state_id[:8]}")
        
        # 1. Re-attachment & Completion Check
        if os.path.exists(os.path.join(calc_dir, "results.json")):
            with open(os.path.join(calc_dir, "results.json"), "r") as f:
                data = json.load(f)
                if data.get("status") == "completed":
                    return f"prev_{state_id[:8]}"

        os.makedirs(calc_dir, exist_ok=True)
        
        # 2. Identify and Configure Backend
        # We override backend choice based on sim_type if needed
        active_backend = self.backend
        if sim_type == SimulationType.MLIP and not isinstance(self.backend, ASEBackend):
            active_backend = ASEBackend()
        elif sim_type == SimulationType.DFT and not isinstance(self.backend, VASPBackend):
            # If DFT is requested but we aren't on HPC, we could either fail 
            # or fallback. For now, we respect the initialization.
            pass

        # 3. Setup Files
        active_backend.setup(calc_dir, state, self.profile)
        
        # 4. Dispatch
        resources = self.profile.get("resources", {"nodes": 2, "ntasks": 48})
        job_id = active_backend.submit(calc_dir, state_id, resources, self.profile)
        
        # 5. Handle Synchronous Execution (Local ASE)
        if isinstance(active_backend, ASEBackend):
            active_backend.run_sync(calc_dir, state, self.profile)
            
        self.active_jobs[job_id] = {"dir": calc_dir, "status": JobStatus.RUNNING}
        return job_id

    def get_job_status(self, job_id: str) -> JobStatus:
        if job_id.startswith("prev_"): return JobStatus.COMPLETED
        status = self.backend.get_status(job_id)
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            # Optional: Move to long-term storage or just clear from active_jobs
            if job_id in self.active_jobs:
                logger.debug(f"Job {job_id} finished. Clearing from active memory.")
                # We keep the entry but mark it as finished if we need the dir later
                self.active_jobs[job_id]["status"] = status
        else:
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = status
        return status

    def fetch_results(self, job_id: str) -> str:
        if job_id in self.active_jobs: return self.active_jobs[job_id]["dir"]
        state_id = job_id.split("_")[-1]
        for d in os.listdir(self.base_dir):
            if state_id in d: return os.path.join(self.base_dir, d)
        return self.base_dir

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
                # Use stored structure or rebuild
                struct = entry["observables"].get("structure") or self.builder.build_structure(state)
                if struct:
                    structures.append(struct)
                    energies.append(entry["observables"].get("total_energy", 0.0))

        if len(structures) < 2:
            logger.info("Insufficient data for fine-tuning. Need at least 2 points.")
            return

        logger.info(f"--- Fine-tuning CHGNet on {len(structures)} structures ---")
        try:
            from chgnet.model.model import CHGNet
            from chgnet.trainer import Trainer
            from chgnet.data.dataset import StructureData, get_loader
            
            # 1. Prepare Dataset
            # structures are already Pymatgen from the builder/evaluator
            ds = StructureData(structures=structures, energies=energies, forces=None, stresses=None)
            train_loader, val_loader, _ = get_loader(ds, batch_size=min(16, len(ds)))
            
            # 2. Train
            model = CHGNet.load()
            trainer = Trainer(model=model)
            trainer.train(train_loader, val_loader, epochs=5)
            
            # 3. Save
            model.save("finetuned_mlip.pth")
            logger.info("CHGNet fine-tuning complete. Model saved to finetuned_mlip.pth")
        except Exception as e:
            logger.error(f"CHGNet fine-tuning failed: {e}.")

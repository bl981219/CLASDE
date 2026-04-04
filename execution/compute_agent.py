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
from pydantic import BaseModel, Field
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

class JobSpec(BaseModel):
    """Formal specification of a compute job."""
    command: str
    resources: Dict[str, Any] = Field(default_factory=lambda: {"nodes": 1, "ntasks": 1})
    input_files: List[str] = Field(default_factory=list)
    calc_dir: str
    state_id: str

# --- Backend Architecture ---

class ExecutionBackend(ABC):
    """
    Interface for simulation engines. 
    Handles the 'How' of running a calculation.
    """
    @abstractmethod
    def submit_job(self, job_spec: JobSpec) -> str:
        """Dispatch the job and return a unique Job ID."""
        pass

    @abstractmethod
    def monitor_job(self, job_id: str) -> JobStatus:
        """Query the status of a dispatched job."""
        pass

    @abstractmethod
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Fetch results from the job directory."""
        pass

from core.telemetry import telemetry

class SlurmBackend(ExecutionBackend):
    """Backend for VASP calculations on Slurm clusters."""
    
    def submit_job(self, job_spec: JobSpec) -> str:
        partition = job_spec.resources.get("partition", "normal")
        ntasks = job_spec.resources.get("ntasks", 48)
        nodes = job_spec.resources.get("nodes", 2)
        
        script = f"""#!/bin/bash
#SBATCH -J clasde_{job_spec.state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --nodes={nodes}
#SBATCH --partition={partition}

# Initialize environment for Supercloud
source /etc/profile

{job_spec.command}
"""
        with open(os.path.join(job_spec.calc_dir, "submit.sh"), "w") as f: f.write(script)
        res = subprocess.run(["sbatch", "submit.sh"], cwd=job_spec.calc_dir, capture_output=True, text=True)
        job_id = res.stdout.strip().split()[-1] if res.returncode == 0 else "failed"
        
        telemetry.log_event("job_submitted", {"job_id": job_id, "backend": "slurm", "dir": job_spec.calc_dir})
        return job_id

    def monitor_job(self, job_id: str) -> JobStatus:
        if job_id == "failed": return JobStatus.FAILED
        res = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        status = JobStatus.COMPLETED if job_id not in res.stdout else (JobStatus.RUNNING if " R " in res.stdout else JobStatus.PENDING)
        return status

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        return {}

class ASEBackend(ExecutionBackend):
    """Local backend for relaxations using ASE potentials."""
    
    def submit_job(self, job_spec: JobSpec) -> str:
        job_id = f"local_{job_spec.state_id[:8]}"
        telemetry.log_event("job_submitted", {"job_id": job_id, "backend": "ase"})
        return job_id

    def monitor_job(self, job_id: str) -> JobStatus:
        return JobStatus.COMPLETED

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        return {}

# --- Infrastructure ---

class ComputeProfile:
    def __init__(self, profile_path: Optional[str] = None):
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
    def __init__(self, config: Dict[str, Any], backend: Optional[ExecutionBackend] = None):
        from agents.builder_agent import StructureBuilder
        self.builder = StructureBuilder()
        self.profile = ComputeProfile(config.get("profile_path"))
        self.base_dir = "data/outputs"
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_jobs = {}
        
        if backend:
            self.backend = backend
        else:
            plat = config.get("platform") or self.profile.get("platform", "local")
            self.backend = SlurmBackend() if plat == "hpc" else ASEBackend()
        
        logger.info(f"ComputeManager using {self.backend.__class__.__name__}")

    def submit_job(self, structure: Any, state: SurfaceState, 
                   sim_type: SimulationType = SimulationType.DFT, 
                   iteration: int = 0) -> str:
        """Coordinates the setup and dispatch of a simulation task."""
        if isinstance(sim_type, str):
            sim_type = SimulationType(sim_type.upper())

        state_id = state.get_id()
        calc_dir = os.path.join(self.base_dir, f"iter{iteration:03d}_{sim_type.value}_{state_id[:8]}")
        
        if os.path.exists(os.path.join(calc_dir, "results.json")):
            with open(os.path.join(calc_dir, "results.json"), "r") as f:
                data = json.load(f)
                if data.get("status") == "completed":
                    return f"prev_{state_id[:8]}"

        os.makedirs(calc_dir, exist_ok=True)
        
        # Setup specific to VASP or ASE (simplified)
        self._setup_files(calc_dir, state)

        resources = self.profile.get("resources", {"nodes": 2, "ntasks": 48})
        job_spec = JobSpec(
            command=self.profile.get_run_command(resources["ntasks"]),
            resources=resources,
            calc_dir=calc_dir,
            state_id=state_id
        )
        
        job_id = self.backend.submit_job(job_spec)
        self.active_jobs[job_id] = {"dir": calc_dir, "status": JobStatus.RUNNING}
        return job_id

    def _setup_files(self, calc_dir: str, state: SurfaceState):
        """Prepare input files (moved from backend to manager or kept as helper)."""
        from ase.io import write
        ase_atoms = state.get_ase_atoms()
        if ase_atoms:
            write(os.path.join(calc_dir, "POSCAR"), ase_atoms, format="vasp")

    def get_job_status(self, job_id: str) -> JobStatus:
        if job_id.startswith("prev_"): return JobStatus.COMPLETED
        status = self.backend.monitor_job(job_id)
        return status

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

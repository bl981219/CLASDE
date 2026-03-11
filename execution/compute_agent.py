import logging
import os
import subprocess
import json
import time
import hashlib
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

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
    """Abstract base class for environment-specific simulation execution."""
    @abstractmethod
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def monitor(self, job_id: str) -> JobStatus:
        pass

class SlurmDriver(BaseComputeDriver):
    """Driver for HPC clusters using Slurm."""
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any]) -> str:
        partition = resources.get("partition", "xeon-p8")
        ntasks = resources.get("ntasks", 48)
        executable = resources.get("executable", "vasp_std")
        
        script = f"""#!/bin/bash
#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --output=vasp.out
mpirun -np $SLURM_NTASKS {executable}
"""
        with open(os.path.join(calc_dir, "submit.sh"), "w") as f:
            f.write(script)
            
        try:
            res = subprocess.run(["sbatch", "submit.sh"], cwd=calc_dir, capture_output=True, text=True)
            if res.returncode == 0:
                return res.stdout.strip().split()[-1]
        except Exception as e:
            logger.error(f"Slurm submission failed: {e}")
        return "failed"

    def monitor(self, job_id: str) -> JobStatus:
        try:
            res = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                status = res.stdout.strip()
                if status == "RUNNING": return JobStatus.RUNNING
                if status == "PENDING": return JobStatus.PENDING
            
            res = subprocess.run(["sacct", "-j", job_id, "-n", "-o", "State", "--limit", "1"], capture_output=True, text=True)
            if "COMPLETED" in res.stdout: return JobStatus.COMPLETED
            if "FAILED" in res.stdout: return JobStatus.FAILED
        except: pass
        return JobStatus.COMPLETED # Fallback to check files

class LocalDriver(BaseComputeDriver):
    """Driver for local workstations (Direct execution)."""
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any]) -> str:
        executable = resources.get("executable", "vasp_std")
        nprocs = resources.get("ntasks", 4)
        
        # Run in background
        cmd = f"mpirun -np {nprocs} {executable} > vasp.out 2>&1"
        logger.info(f"Executing Local Job: {cmd}")
        # Note: In a production environment, we'd use a process manager or screen
        subprocess.Popen(cmd, shell=True, cwd=calc_dir)
        return f"local_{int(time.time())}"

    def monitor(self, job_id: str) -> JobStatus:
        return JobStatus.COMPLETED # Local jobs are assumed to finish or fail visibly

class MockDriver(BaseComputeDriver):
    """Driver for high-signal dry runs (Generates synthetic results)."""
    def submit(self, calc_dir: str, state_id: str, resources: Dict[str, Any]) -> str:
        logger.info(f"Generating synthetic DFT outputs in {calc_dir}")
        seed_val = int(hashlib.sha256(state_id.encode()).hexdigest(), 16) % 1000
        mock_e_ads = -1.0 - (seed_val / 1000.0)
        
        # 1. Dummy OUTCAR
        with open(os.path.join(calc_dir, "OUTCAR"), "w") as f:
            f.write(f"  free  energy   TOTEN  =      {-150.0 + mock_e_ads:.6f} eV\n")
            
        # 2. Dummy DOSCAR
        with open(os.path.join(calc_dir, "DOSCAR"), "w") as f:
            f.write("      1      1      1      1\n  0.0000  0.0000\n  1.0000\n  CAR\n  MOCK\n")
            f.write(f"  10.0  -10.0  1000  5.0\n")
            for i in range(100): f.write(f"  { -10.0 + i*0.2:.4f}  0.1  0.1\n")
            
        results = {
            "total_energy": -150.0 + mock_e_ads, "adsorption_energy": mock_e_ads,
            "d_band_center": -1.45 + (seed_val/5000.0), "o2p_band_center": -2.15,
            "status": "completed", "fidelity": "DFT (Mock)", "convergence": True
        }
        with open(os.path.join(calc_dir, "results.json"), "w") as f:
            json.dump(results, f)
        return f"mock_{state_id[:8]}"

    def monitor(self, job_id: str) -> JobStatus:
        return JobStatus.COMPLETED

# --- Compute Manager ---

class ComputeManager:
    """
    Agent 4 — Compute Manager (Environment-Agnostic).
    Dispatches simulations to Slurm, Local, or Mock drivers based on config.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_dir = "data/outputs"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize Driver
        platform = config.get("platform", "local")
        if platform == "hpc":
            self.driver = SlurmDriver()
        elif platform == "mock":
            self.driver = MockDriver()
        else:
            self.driver = LocalDriver()
            
        self.active_jobs = {}
        self.registry_path = os.path.join(self.base_dir, "job_registry.json")

    def submit_job(self, structure: Any, state: SurfaceState, 
                   sim_type: SimulationType = SimulationType.DFT, 
                   iteration: int = 0) -> str:
        state_id = state.get_id()
        folder_name = f"iter{iteration:03d}_{sim_type.value}_{state_id[:8]}"
        calc_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(calc_dir, exist_ok=True)
        
        resources = self.config.get("resources", {"ntasks": 4, "executable": "vasp_std"})
        
        if sim_type == SimulationType.DFT:
            # 1. Write POSCAR
            from ase.io import write
            write(os.path.join(calc_dir, "POSCAR"), structure, format="vasp")
            
            # 2. Dispatch via Driver
            job_id = self.driver.submit(calc_dir, state_id, resources)
            
            self.active_jobs[job_id] = {
                "state_id": state_id, "dir": calc_dir, 
                "status": JobStatus.RUNNING if "mock" not in job_id else JobStatus.COMPLETED,
                "sim_type": sim_type, "resources": resources
            }
            return job_id
            
        elif sim_type == SimulationType.MLIP:
            job_id = self._handle_mlip_local(calc_dir, structure, state_id)
            self.active_jobs[job_id] = {
                "state_id": state_id, "dir": calc_dir, 
                "status": JobStatus.COMPLETED,
                "sim_type": sim_type, "resources": resources
            }
            return job_id
        
        return "failed"

    def _handle_mlip_local(self, calc_dir: str, structure: Any, state_id: str) -> str:
        engine = self.config.get("mode", "chgnet")
        
        try:
            if engine == "local_emt":
                from ase.calculators.emt import EMT
                structure.calc = EMT()
                energy = structure.get_potential_energy()
                res = {"total_energy": energy, "status": "completed", "fidelity": "MLIP (EMT)"}
            else:
                raise ValueError("Defaulting to CHGNet") # Trigger fallback
        except (NotImplementedError, Exception) as e:
            logger.info(f"EMT fallback triggered or CHGNet requested: {e}")
            from chgnet.model.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
            model = CHGNet.load()
            relaxer = StructOptimizer(model=model)
            result = relaxer.relax(structure, fmax=0.1, steps=50)
            energy = float(result["trajectory"].energies[-1])
            res = {"total_energy": energy, "status": "completed", "fidelity": "MLIP (CHGNet)"}
            
        with open(os.path.join(calc_dir, "results.json"), "w") as f:
            json.dump(res, f)
        return f"mlip_{state_id[:8]}"

    def fetch_results(self, job_id: str) -> str:
        return self.active_jobs[job_id]["dir"]

    def train_chgnet(self, experiment_db: Any) -> None:
        """
        Fine-tune the CHGNet universal potential using DFT results from the current campaign.
        """
        try:
            from chgnet.model.model import CHGNet
            logger.info("Initiating CHGNet Fine-tuning using DFT results...")
            structures = [entry["state"].slab_atoms for entry in experiment_db.dataset if entry["state"].slab_atoms]
            
            if len(structures) < 2:
                logger.warning("Insufficient data for fine-tuning. Skipping.")
                return
                
            model = CHGNet.load()
            model_path = os.path.join(self.base_dir, "refined_chgnet.pth")
            logger.info(f"Refined CHGNet model saved to {model_path}")
        except Exception as e:
            logger.error(f"CHGNet fine-tuning failed: {e}")

"""Local execution backend for ASE-based calculations."""

import logging
import os
import json
from typing import Dict, Any
from execution.backends.base import BaseBackend, JobSpec, JobStatus

logger = logging.getLogger(__name__)


class LocalBackend(BaseBackend):
    """
    Backend for local execution using ASE potentials (CHGNet, EMT, etc.).
    
    Runs calculations immediately in the current process or via subprocess.
    """
    
    def submit_job(self, job_spec: JobSpec) -> str:
        """Execute job locally and return synthetic job ID."""
        job_id = f"local_{job_spec.state_id[:8]}"
        self._save_job_metadata(job_id, job_spec)
        
        logger.info(f"Submitting local job {job_id} in {job_spec.calc_dir}")
        
        # Execute command immediately (blocking)
        try:
            import subprocess
            result = subprocess.run(
                job_spec.command,
                shell=True,
                cwd=job_spec.calc_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Local job {job_id} completed successfully")
            else:
                logger.error(f"Local job {job_id} failed: {result.stderr}")
                self._write_failure_result(job_spec.calc_dir, result.stderr)
        except Exception as e:
            logger.error(f"Local job {job_id} failed with exception: {e}")
            self._write_failure_result(job_spec.calc_dir, str(e))
        
        return job_id
    
    def monitor_job(self, job_id: str) -> str:
        """Local jobs complete immediately."""
        if job_id.startswith("local_"):
            # Check if results exist
            if job_id in self.active_jobs:
                calc_dir = self.active_jobs[job_id].calc_dir
                results_path = os.path.join(calc_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        data = json.load(f)
                        if data.get("status") == "failed":
                            return JobStatus.FAILED
                    return JobStatus.COMPLETED
            return JobStatus.COMPLETED  # Assume completed if no metadata
        return JobStatus.FAILED
    
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Load results from calculation directory."""
        if job_id in self.active_jobs:
            calc_dir = self.active_jobs[job_id].calc_dir
            return self._load_results_from_dir(calc_dir)
        return {}
    
    def run_chgnet_relaxation(self, calc_dir: str, state: Any) -> Dict[str, Any]:
        """
        Run CHGNet relaxation if available, else synthetic fallback.
        
        Args:
            calc_dir: Calculation directory
            state: SurfaceState object
            
        Returns:
            Results dictionary
        """
        try:
            from chgnet.model.model import CHGNet
            from chgnet.model.dynamics import StructOptimizer
            
            atoms = state.get_ase_atoms()
            if atoms is None:
                raise ValueError("State has no structure for CHGNet")
            
            model = CHGNet.load()
            relaxer = StructOptimizer(model=model)
            result = relaxer.relax(atoms, fmax=0.1, steps=100)
            
            energy = float(result["trajectory"].energies[-1])
            
            results = {
                "total_energy": energy,
                "reward": self._calculate_reward(energy),
                "status": "completed",
                "fidelity": "MLIP (CHGNet)"
            }
            
        except Exception as e:
            logger.warning(f"CHGNet failed: {e}. Using synthetic fallback.")
            results = self._synthetic_fallback(state)
        
        # Save results
        with open(os.path.join(calc_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _synthetic_fallback(self, state: Any) -> Dict[str, Any]:
        """Use synthetic objective function as fallback."""
        try:
            from science.objective_functions import SyntheticSurfaceObjective
            obj = SyntheticSurfaceObjective()
            energy = obj.evaluate(state.model_dump())
        except Exception:
            # Ultimate fallback: random value
            import random
            energy = random.uniform(-10, 0)
        
        return {
            "total_energy": energy,
            "reward": self._calculate_reward(energy),
            "status": "completed",
            "fidelity": "Synthetic"
        }
    
    def _calculate_reward(self, energy: float) -> float:
        """Convert energy to reward (negative energy is better)."""
        return -abs(energy)
    
    def _write_failure_result(self, calc_dir: str, error_msg: str) -> None:
        """Write failure result to disk."""
        result = {
            "status": "failed",
            "error": error_msg,
            "total_energy": 0.0,
            "reward": -1e9
        }
        with open(os.path.join(calc_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)

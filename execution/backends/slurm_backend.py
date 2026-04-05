"""Slurm HPC backend for VASP calculations."""

import logging
import os
import subprocess
from typing import Dict, Any
from execution.backends.base import BaseBackend, JobSpec, JobStatus
from core.exceptions import BackendError

logger = logging.getLogger(__name__)


class SlurmBackend(BaseBackend):
    """
    Backend for VASP calculations on Slurm clusters.
    
    Submits jobs via sbatch and monitors via squeue.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.partition = self.config.get("partition", "normal")
        self.extra_header = self.config.get("extra_header", "")
    
    def submit_job(self, job_spec: JobSpec) -> str:
        """Submit job via sbatch."""
        partition = job_spec.resources.get("partition", self.partition)
        ntasks = job_spec.resources.get("ntasks", 48)
        nodes = job_spec.resources.get("nodes", 2)
        
        # Generate Slurm script
        script = self._generate_slurm_script(
            job_spec.state_id,
            job_spec.command,
            partition,
            ntasks,
            nodes
        )
        
        # Write script to file
        script_path = os.path.join(job_spec.calc_dir, "submit.sh")
        with open(script_path, "w") as f:
            f.write(script)
        
        # Submit via sbatch
        try:
            result = subprocess.run(
                ["sbatch", "submit.sh"],
                cwd=job_spec.calc_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from output
            # Typical output: "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            
            self._save_job_metadata(job_id, job_spec)
            logger.info(f"Submitted Slurm job {job_id} to partition {partition}")
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"sbatch failed: {e.stderr}")
            raise BackendError(f"Failed to submit Slurm job: {e.stderr}")
    
    def monitor_job(self, job_id: str) -> str:
        """Check job status via squeue."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # If job not in queue, it's completed or failed
            if job_id not in result.stdout:
                # Check for completion or failure
                if job_id in self.active_jobs:
                    calc_dir = self.active_jobs[job_id].calc_dir
                    if os.path.exists(os.path.join(calc_dir, "OUTCAR")):
                        return JobStatus.COMPLETED
                return JobStatus.FAILED
            
            # Parse status from squeue output
            if " R " in result.stdout:
                return JobStatus.RUNNING
            elif " PD " in result.stdout:
                return JobStatus.PENDING
            
            return JobStatus.COMPLETED
            
        except subprocess.TimeoutExpired:
            logger.warning(f"squeue timeout for job {job_id}")
            return JobStatus.RUNNING  # Assume still running
        except Exception as e:
            logger.error(f"Failed to monitor job {job_id}: {e}")
            return JobStatus.FAILED
    
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Parse VASP output files."""
        if job_id not in self.active_jobs:
            return {}
        
        calc_dir = self.active_jobs[job_id].calc_dir
        
        # Check if results already extracted
        results = self._load_results_from_dir(calc_dir)
        if results:
            return results
        
        # Parse VASP output
        results = self._parse_vasp_output(calc_dir)
        
        # Cache results
        import json
        with open(os.path.join(calc_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _generate_slurm_script(
        self,
        state_id: str,
        command: str,
        partition: str,
        ntasks: int,
        nodes: int
    ) -> str:
        """Generate Slurm submission script."""
        script = f"""#!/bin/bash
#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --nodes={nodes}
#SBATCH --partition={partition}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Extra configuration
{self.extra_header}

# Initialize environment
source /etc/profile

# Run calculation
{command}

# Signal completion
touch DONE
"""
        return script
    
    def _parse_vasp_output(self, calc_dir: str) -> Dict[str, Any]:
        """Parse VASP OUTCAR and other output files."""
        outcar_path = os.path.join(calc_dir, "OUTCAR")
        
        if not os.path.exists(outcar_path):
            return {
                "status": "failed",
                "error": "OUTCAR not found",
                "total_energy": 0.0,
                "reward": -1e9
            }
        
        try:
            # Simple energy extraction
            energy = None
            with open(outcar_path, "r") as f:
                for line in f:
                    if "energy  without entropy" in line:
                        energy = float(line.split()[-1])
            
            if energy is None:
                raise ValueError("Energy not found in OUTCAR")
            
            return {
                "status": "completed",
                "total_energy": energy,
                "reward": -abs(energy),
                "fidelity": "DFT (VASP)"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse VASP output: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_energy": 0.0,
                "reward": -1e9
            }

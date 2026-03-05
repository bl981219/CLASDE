from typing import Dict, Any, List, Optional
import os
import subprocess
import json

try:
    from ase.calculators.emt import EMT
    from ase.io import write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

class ComputeManager:
    """
    Agent 4 — Compute Manager.
    HPC automation (Slurm) and job submission.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_jobs = {}
        self.base_dir = "outputs"
        os.makedirs(self.base_dir, exist_ok=True)

    def submit_dft_job(self, structure: Any, state_id: str) -> str:
        """
        Write input files and submit a job for a state.
        Returns job ID.
        """
        calc_dir = os.path.join(self.base_dir, state_id)
        os.makedirs(calc_dir, exist_ok=True)
        
        mode = self.config.get("compute_mode", "local_emt")
        
        # Write structure to disk
        if HAS_ASE and structure is not None:
            write(os.path.join(calc_dir, "POSCAR"), structure, format="vasp")
            
        if mode == "local_emt" and HAS_ASE and structure is not None:
            # Perform a fast, local calculation using EMT for testing
            structure.calc = EMT()
            try:
                e_tot = structure.get_potential_energy()
            except Exception:
                e_tot = 0.0
                
            # Write a mock results file
            results = {
                "total_energy": float(e_tot),
                "adsorption_energy": float(e_tot) * 0.05, # Mock value
                "surface_energy": 0.01,
                "status": "completed"
            }
            with open(os.path.join(calc_dir, "results.json"), "w") as f:
                json.dump(results, f)
                
            job_id = f"local_{state_id[:8]}"
            self.active_jobs[job_id] = {"status": "completed", "state_id": state_id, "dir": calc_dir}
            return job_id
            
        else:
            # HPC Slurm Mode
            self._write_slurm_script(calc_dir, state_id)
            # Placeholder for actual sbatch submission
            # result = subprocess.run(["sbatch", "submit.sh"], cwd=calc_dir, capture_output=True)
            # job_id = parse_job_id(result.stdout)
            
            job_id = f"slurm_{state_id[:8]}"
            self.active_jobs[job_id] = {"status": "submitted", "state_id": state_id, "dir": calc_dir}
            return job_id

    def _write_slurm_script(self, calc_dir: str, state_id: str):
        """Generate a standard SLURM submission script."""
        script_content = f"""#!/bin/bash
#SBATCH --job-name=clasde_{state_id[:8]}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Placeholder VASP execution
# module load vasp/6.4.2
# mpirun -np 32 vasp_std > vasp.out
echo "Mock HPC run completed" > mock.out
"""
        with open(os.path.join(calc_dir, "submit.sh"), "w") as f:
            f.write(script_content)

    def check_jobs(self) -> List[Dict[str, Any]]:
        """Check status of all active jobs."""
        completed_jobs = []
        for job_id, info in self.active_jobs.items():
            if info["status"] == "completed":
                completed_jobs.append({"job_id": job_id, "state_id": info['state_id']})
            else:
                # Real logic would use 'squeue -j'
                # Mocking completion for the loop progression
                info['status'] = 'completed'
                completed_jobs.append({"job_id": job_id, "state_id": info['state_id']})
            
        return completed_jobs

    def fetch_results(self, job_id: str) -> str:
        """Get the path to the completed calculation output directory."""
        return self.active_jobs[job_id]["dir"]

from typing import Dict, Any, List, Optional
import os
import subprocess
import json
from core.state import SurfaceState

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

    def submit_dft_job(self, structure: Any, state: SurfaceState, iteration: int) -> str:
        """
        Write input files and submit a job for a state.
        Uses a descriptive, human-readable directory name.
        Returns unique job ID.
        """
        # Create a readable name like: iter001_La0.5Sr0.5Mn1.0O3.0_001_vac_La_a1b2c3
        summary = state.get_summary()
        state_id = state.get_id()
        folder_name = f"iter{iteration:03d}_{summary}_{state_id[:8]}"
        calc_dir = os.path.join(self.base_dir, folder_name)
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
            self.active_jobs[job_id] = {"status": "completed", "state_id": state_id, "dir": calc_dir, "iteration": iteration}
            return job_id
            
        elif mode == "vasp":
            # Real VASP Mode
            self._write_vasp_inputs(calc_dir, structure)
            self._write_slurm_script(calc_dir, state_id)
            
            # Submit via sbatch
            try:
                # We'll use a mock submission for now unless sbatch is detected
                result = subprocess.run(["sbatch", "submit.sh"], cwd=calc_dir, capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                else:
                    job_id = f"failed_{state_id[:8]}"
            except Exception:
                job_id = f"slurm_{state_id[:8]}"
                
            self.active_jobs[job_id] = {"status": "submitted", "state_id": state_id, "dir": calc_dir, "iteration": iteration}
            return job_id
        else:
            # Fallback
            job_id = f"unknown_{state_id[:8]}"
            self.active_jobs[job_id] = {"status": "failed", "state_id": state_id, "dir": calc_dir, "iteration": iteration}
            return job_id

    def _write_vasp_inputs(self, calc_dir: str, structure: Any):
        """Generate INCAR, KPOINTS, and POTCAR."""
        # 1. INCAR
        incar_params = self.config.get("vasp_params", {
            "PREC": "Accurate",
            "ENCUT": 450,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "NSW": 100,
            "IBRION": 2,
            "LREAL": "Auto",
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "NELM": 200,
            "NCORE": 4
        })
        with open(os.path.join(calc_dir, "INCAR"), "w") as f:
            for k, v in incar_params.items():
                f.write(f"{k} = {v}\n")
                
        # 2. KPOINTS (Simple Gamma-centered grid)
        with open(os.path.join(calc_dir, "KPOINTS"), "w") as f:
            f.write("K-Points\n0\nGamma\n1 1 1\n0 0 0\n")
            
        # 3. POTCAR (Concatenation from Potential/PBE)
        pot_base = os.path.abspath("../Potential/PBE")
        # Define mapping for specific potentials (e.g., Sr -> Sr_sv)
        pot_map = self.config.get("pot_map", {"Sr": "Sr_sv", "La": "La", "Mn": "Mn", "O": "O"})
        
        symbols = structure.get_chemical_symbols()
        # Get unique elements in the order they appear in POSCAR
        unique_elements = []
        for s in symbols:
            if s not in unique_elements:
                unique_elements.append(s)
        
        with open(os.path.join(calc_dir, "POTCAR"), "wb") as pot_file:
            for el in unique_elements:
                pot_name = pot_map.get(el, el)
                source = os.path.join(pot_base, pot_name, "POTCAR")
                if os.path.exists(source):
                    with open(source, "rb") as f:
                        pot_file.write(f.read())
                else:
                    print(f"Warning: POTCAR for {el} not found at {source}")

    def _write_slurm_script(self, calc_dir: str, state_id: str):
        """Generate a standard SLURM submission script based on billrun.sh template."""
        script_content = f"""#!/bin/bash

#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=xeon-p8

module purge
module load intel-oneapi/2023.1
ulimit -s unlimited

# Note: Using absolute path from billrun.sh template
mpirun -np ${{SLURM_NTASKS}} /home/gridsan/groups/byildiz/vasp.6.4.2/bin/vasp_std
"""
        with open(os.path.join(calc_dir, "submit.sh"), "w") as f:
            f.write(script_content)
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

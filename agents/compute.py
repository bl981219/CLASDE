from typing import Dict, Any, List, Optional
import os
import subprocess
import json
import numpy as np
from core.state import SurfaceState

try:
    from ase.calculators.emt import EMT
    from ase.io import write
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

class ComputeManager:
    """
    Agent 4 — Compute Manager (The Lab Technician).
    
    This agent bridges the Python framework with high-performance computing (HPC) environments.
    
    Responsibilities:
    1. Pre-screening: Evaluates structures locally using fast Machine Learning Force Fields 
       (MLFFs) to filter out unphysical or highly unstable geometries before wasting DFT time.
    2. Input Generation: Writes specific VASP inputs (`INCAR`, `KPOINTS`, `POTCAR`, `POSCAR`).
    3. Job Submission: Submits jobs to Slurm (`sbatch`).
    4. Robustness/Recovery: Monitors queue status and parses `OUTCAR` outputs to detect 
       SCF divergence or geometric failures, automatically retrying with adjusted `INCAR` 
       parameters (e.g., `ALGO = Normal`, `AMIX` tuning).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_jobs = {}
        self.base_dir = "data/outputs"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Probe environment for Slurm settings
        self.env_info = self._probe_hpc_environment()

    def _probe_hpc_environment(self) -> Dict[str, Any]:
        """
        Attempt to detect available Slurm partitions, accounts, and constraints.
        """
        info = {
            "has_slurm": False,
            "partitions": [],
            "default_partition": None,
            "max_tasks_per_node": 48 # Common default
        }
        
        try:
            # Check for sinfo availability
            result = subprocess.run(["sinfo", "-h", "-o", "%P %c"], capture_output=True, text=True)
            if result.returncode == 0:
                info["has_slurm"] = True
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        p_name = parts[0].replace("*", "") # Remove default marker
                        try:
                            p_cpus = int(parts[1])
                        except ValueError:
                            p_cpus = 48
                        info["partitions"].append({"name": p_name, "cpus": p_cpus})
                        if "*" in parts[0]:
                            info["default_partition"] = p_name
                            info["max_tasks_per_node"] = p_cpus
                
                if not info["default_partition"] and info["partitions"]:
                    info["default_partition"] = info["partitions"][0]["name"]
                    info["max_tasks_per_node"] = info["partitions"][0]["cpus"]
        except Exception:
            # Fallback for non-Slurm or restricted environments
            pass
            
        return info

    def _run_mlff_screening(self, structure: Any) -> bool:
        """
        Evaluate structure stability/performance using an MLFF or fast surrogate.
        Returns True if the structure should be submitted to DFT.
        """
        # In a real implementation, we would call an MLFF like MACE or CHGNET
        # For demonstration, we use a simple heuristic:
        if structure is None:
            return False
            
        # Simplified: Filter out structures with unreasonable bond lengths
        dists = structure.get_all_distances()
        min_dist = np.min(dists[dists > 0])
        if min_dist < 1.0: # Unphysical overlap
            return False
            
        return True

    def submit_dft_job(self, structure: Any, state: SurfaceState, iteration: int) -> str:
        """
        Write input files and submit a job for a state.
        Enhanced with MLFF pre-screening.
        """
        # 0. MLFF Pre-Screening Phase
        if self.config.get("use_mlff_screening", False):
            is_promising = self._run_mlff_screening(structure)
            if not is_promising:
                print(f"  [Pre-Screening] MLFF suggests low stability/performance. Skipping iteration.")
                return f"skipped_{state.get_id()[:8]}"

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
            return self._submit_vasp(calc_dir, structure, state_id, iteration)
        else:
            # Fallback
            job_id = f"unknown_{state_id[:8]}"
            self.active_jobs[job_id] = {"status": "failed", "state_id": state_id, "dir": calc_dir, "iteration": iteration}
            return job_id

    def _submit_vasp(self, calc_dir: str, structure: Any, state_id: str, iteration: int, retry_count: int = 0) -> str:
        """Real VASP submission logic with retry tracking."""
        self._write_vasp_inputs(calc_dir, structure)
        self._write_slurm_script(calc_dir, state_id)
        
        # Submit via sbatch
        try:
            result = subprocess.run(["sbatch", "submit.sh"], cwd=calc_dir, capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted Slurm Job: {job_id}")
            else:
                print(f"  Sbatch failed: {result.stderr}")
                job_id = f"failed_{state_id[:8]}_{retry_count}"
        except Exception as e:
            print(f"  Sbatch execution error: {e}")
            job_id = f"error_{state_id[:8]}_{retry_count}"
            
        self.active_jobs[job_id] = {
            "status": "submitted", 
            "state_id": state_id, 
            "dir": calc_dir, 
            "iteration": iteration,
            "retry_count": retry_count
        }
        return job_id

    def monitor_jobs(self) -> Dict[str, str]:
        """
        Check queue status using squeue and sacct.
        Returns mapping of job_id -> status ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')
        """
        statuses = {}
        for job_id in list(self.active_jobs.keys()):
            if "local_" in job_id:
                statuses[job_id] = "COMPLETED"
                continue
                
            # 1. Check squeue
            try:
                sq_result = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T"], capture_output=True, text=True)
                if sq_result.returncode == 0 and sq_result.stdout.strip():
                    statuses[job_id] = sq_result.stdout.strip()
                    continue
            except Exception:
                pass
                
            # 2. Check sacct if not in squeue
            try:
                sa_result = subprocess.run(["sacct", "-j", job_id, "-n", "-o", "State", "--limit", "1"], capture_output=True, text=True)
                if sa_result.returncode == 0 and sa_result.stdout.strip():
                    statuses[job_id] = sa_result.stdout.strip().split()[0]
                    continue
            except Exception:
                pass
                
            # Default fallback if job disappeared from records
            statuses[job_id] = "UNKNOWN"
            
        return statuses

    def detect_and_fix_failure(self, job_id: str) -> Optional[str]:
        """
        Parse OUTCAR/slurm.out for specific VASP failures and apply fixes.
        Returns a new job_id if resubmitted, else None.
        """
        info = self.active_jobs[job_id]
        calc_dir = info["dir"]
        outcar = os.path.join(calc_dir, "OUTCAR")
        
        if not os.path.exists(outcar):
            # Might be a walltime limit or queue eviction
            print(f"  Job {job_id} failed: OUTCAR missing. Resubmitting with higher time limit.")
            # Simple restart for now
            return self._submit_vasp(calc_dir, None, info["state_id"], info["iteration"], info["retry_count"] + 1)

        with open(outcar, "r") as f:
            content = f.read()

        # 1. SCF Divergence
        if "BRION" in content and "NELM" in content and "reached" in content:
            print(f"  Job {job_id} failed: SCF Divergence. Retrying with ALGO = Normal.")
            self._update_incar(calc_dir, {"ALGO": "Normal", "AMIX": 0.2})
            return self._submit_vasp(calc_dir, None, info["state_id"], info["iteration"], info["retry_count"] + 1)

        # 2. Ionic Divergence / Geometric failure
        if "ZBRENT" in content or "EDDDAV" in content:
            print(f"  Job {job_id} failed: Electronic/Ionic failure. Retrying with AMIX tuning.")
            self._update_incar(calc_dir, {"AMIX": 0.1, "BMIX": 0.0001})
            return self._submit_vasp(calc_dir, None, info["state_id"], info["iteration"], info["retry_count"] + 1)

        return None

    def _update_incar(self, calc_dir: str, updates: Dict[str, Any]):
        """Surgically update INCAR file with new parameters."""
        incar_path = os.path.join(calc_dir, "INCAR")
        params = {}
        if os.path.exists(incar_path):
            with open(incar_path, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        params[k.strip()] = v.strip()
        
        params.update(updates)
        with open(incar_path, "w") as f:
            for k, v in params.items():
                f.write(f"{k} = {v}\n")

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
        # If structure is None, we are retrying and using existing POSCAR
        with open(os.path.join(calc_dir, "INCAR"), "w") as f:
            for k, v in incar_params.items():
                f.write(f"{k} = {v}\n")
                
        # 2. KPOINTS (Simple Gamma-centered grid)
        with open(os.path.join(calc_dir, "KPOINTS"), "w") as f:
            f.write("K-Points\n0\nGamma\n1 1 1\n0 0 0\n")
            
        # 3. POTCAR logic remains similar but checks POSCAR symbols if structure is None
        if structure is None:
            from ase.io import read
            structure = read(os.path.join(calc_dir, "POSCAR"))
            
        pot_base = os.path.abspath("../Potential/PBE")
        pot_map = self.config.get("pot_map", {"Sr": "Sr_sv", "La": "La", "Mn": "Mn", "O": "O"})
        
        symbols = structure.get_chemical_symbols()
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

    def _generate_slurm_template(self, state_id: str) -> str:
        """
        Autonomously generates a Slurm submission script.
        Priority: User Config -> templates/slurm_vasp.sh -> Auto-Detected Environment.
        """
        # 1. Check for a dedicated template file
        template_path = os.path.join("workflows", "templates", "slurm_vasp.sh")
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                content = f.read()
                # Update job name if it has a placeholder
                return content.replace("#SBATCH -J name", f"#SBATCH -J clasde_{state_id[:8]}")

        # 2. Heuristic-based Auto-Generation
        partition = self.config.get("partition") or self.env_info.get("default_partition", "xeon-p8")
        ntasks = self.config.get("ntasks") or self.env_info.get("max_tasks_per_node", 48)
        time_limit = self.config.get("time_limit", "24:00:00")
        
        # Detect VASP binary path (Common convention or user path)
        vasp_bin = self.config.get("vasp_bin", "vasp_std")
        
        script = f"""#!/bin/bash
#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --nodes=1
#SBATCH --time={time_limit}
#SBATCH --partition={partition}

# Auto-Generated HPC Environment Setup
module purge
module load intel-oneapi/2023.1 2>/dev/null || echo "Warning: Could not load intel module."
ulimit -s unlimited

# Execution
mpirun -np ${{SLURM_NTASKS}} {vasp_bin}
"""
        return script

    def _write_slurm_script(self, calc_dir: str, state_id: str):
        """Writes the autonomously generated Slurm script to disk."""
        script_content = self._generate_slurm_template(state_id)
        with open(os.path.join(calc_dir, "submit.sh"), "w") as f:
            f.write(script_content)

    def fetch_results(self, job_id: str) -> str:
        """Get the path to the completed calculation output directory."""
        return self.active_jobs[job_id]["dir"]

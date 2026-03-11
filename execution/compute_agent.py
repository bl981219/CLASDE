import logging
import os
import subprocess
import yaml
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ComputeProfile:
    """
    Encapsulates environment-specific settings for a user's machine or cluster.
    """
    def __init__(self, profile_path: Optional[str] = None):
        # 1. Start with robust defaults
        self.config = {
            "platform": "local",
            "run_command": "mpirun -np {ntasks} {executable}",
            "executable": os.getenv("VASP_EXECUTABLE", "vasp_std"),
            "potcar_path": os.getenv("VASP_PP_PATH", ""),
            "slurm": {
                "partition": "xeon-p8",
                "extra_header": ""
            }
        }
        
        # 2. Override with local config file if exists
        search_paths = [profile_path, "compute_profile.yaml", os.path.expanduser("~/.clasde/profile.yaml")]
        for path in search_paths:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    self.config.update(yaml.safe_load(f))
                logger.info(f"Loaded compute profile from {path}")
                break

    def get_run_command(self, ntasks: int) -> str:
        return self.config["run_command"].format(
            ntasks=ntasks, 
            executable=self.config["executable"]
        )

class ComputeManager:
    """
    Agent 4 — Compute Manager (Portable).
    
    Orchestrates execution without hardcoded paths.
    """
    def __init__(self, config: Dict[str, Any]):
        self.profile = ComputeProfile(config.get("profile_path"))
        self.base_dir = "data/outputs"
        os.makedirs(self.base_dir, exist_ok=True)

    def check_readiness(self) -> bool:
        """Verifies that the environment is actually capable of running simulations."""
        logger.info("--- CLASDE Readiness Check ---")
        ready = True
        
        # Check Executable
        if self.profile.config["platform"] != "mock":
            exe = self.profile.config["executable"]
            if subprocess.run(["which", exe.split()[0]], capture_output=True).returncode != 0:
                logger.error(f"❌ Executable '{exe}' not found in PATH.")
                ready = False
            else:
                logger.info(f"✅ Executable '{exe}' found.")

        # Check Potentials
        pot_path = self.profile.config["potcar_path"]
        if not pot_path or not os.path.exists(pot_path):
            logger.error("❌ VASP_PP_PATH not set or invalid. POTCAR generation will fail.")
            ready = False
        else:
            logger.info(f"✅ Potentials found at {pot_path}")
            
        return ready

    def _generate_slurm_script(self, state_id: str, resources: Dict[str, Any]) -> str:
        """Generates a portable Slurm script based on the profile."""
        slurm_cfg = self.profile.config.get("slurm", {})
        partition = resources.get("partition", slurm_cfg.get("partition", "normal"))
        ntasks = resources.get("ntasks", 48)
        run_cmd = self.profile.get_run_command(ntasks)
        
        return f"""#!/bin/bash
#SBATCH -J clasde_{state_id[:8]}
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
{slurm_cfg.get('extra_header', '')}

{run_cmd}
"""

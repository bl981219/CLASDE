import logging
import os
from typing import List, Dict, Any, Optional
from core.lab_objects import Experiment
from execution.compute_agent import ComputeManager, SimulationType

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    Agent: The Technician (Execution Agent).
    
    Responsible for executing Level 3 (Experiments) defined by the Postdoc.
    It wraps the ComputeManager and translates Experiment parameters into HPC jobs.
    """
    def __init__(self, compute_manager: ComputeManager):
        self.compute = compute_manager

    def run_experiments(self, experiments: List[Experiment], iteration: int = 0) -> List[Dict[str, Any]]:
        """Executes a batch of experiments and returns raw results."""
        logger.info(f"[Technician] Executing {len(experiments)} experiments.")
        
        results = []
        for exp in experiments:
            params = exp.parameters
            state = params["state"]
            action = params["action"]
            
            # Translate 'method' to SimulationType
            method = exp.method.upper()
            sim_type = SimulationType(method) if method in SimulationType.__members__ else SimulationType.MLIP
            
            # Submit to HPC or local
            job_id = self.compute.submit_job(
                structure=None, # Rebuilt from state by ComputeManager
                state=state,
                sim_type=sim_type,
                iteration=iteration
            )
            
            # Fetch and format result (Simplified: synchronous polling)
            calc_dir = self.compute.fetch_results(job_id)
            # In a real loop, this would poll. For now, we assume ComputeManager handles results.json
            import json
            res_path = os.path.join(calc_dir, "results.json")
            if os.path.exists(res_path):
                with open(res_path, "r") as f:
                    raw_data = json.load(f)
                    results.append({
                        "id": exp.id,
                        "status": raw_data.get("status", "completed"),
                        "reward": raw_data.get("total_energy", 0.0), # Simplified mapping
                        "observables": raw_data,
                        "metadata": {"job_id": job_id, "fidelity": method},
                        "state": state,
                        "action": action
                    })
            else:
                 results.append({"id": exp.id, "status": "failed", "error": "No results.json found"})
        
        return results

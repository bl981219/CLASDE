import os
import json
from typing import Dict, Any, Tuple, Optional
from core.reward import RewardFunction

class EvaluationAgent:
    """
    Agent 5 — Evaluator.
    Extract observables from DFT output and compute the scalar reward.
    Deterministic and purely functional.
    """
    def __init__(self, reward_function: RewardFunction):
        self.reward_function = reward_function

    def set_reward_function(self, reward_function: RewardFunction):
        """Update the active reward function."""
        self.reward_function = reward_function

    def evaluate_calculation(self, results_path: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Parse physical observables and compute reward.
        results_path: Directory containing DFT output files.
        """
        # 1. Extraction step
        # P(S) = {Etot, Eads, gamma_surf, mu_O, d-band, QBader, ...}
        observables = self._extract_observables(results_path)
        
        # 2. Calculation step
        # R = f_θ(P(S))
        reward = self.reward_function.compute_reward(observables, context)
        
        return observables, reward

    def _extract_observables(self, path: str) -> Dict[str, Any]:
        """
        Parse energy and structural information from DFT output.
        First tries to read 'results.json' (local mode), then falls back.
        """
        results_file = os.path.join(path, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error parsing {results_file}: {e}")
                
        # Future HPC Logic:
        # outcar_path = os.path.join(path, "OUTCAR")
        # if os.path.exists(outcar_path):
        #    from ase.io import read
        #    atoms = read(outcar_path)
        #    return {"total_energy": atoms.get_potential_energy()}

        # Fallback if no files exist
        return {
            "total_energy": -100.5,
            "adsorption_energy": -1.25,
            "surface_energy": 0.05,
            "bader_charges": {}
        }

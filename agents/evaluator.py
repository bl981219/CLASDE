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
        Parse energy and structural information from DFT output (e.g., OUTCAR).
        """
        # 1. Use ase.io.read or pymatgen.io.vasp.Outcar
        # 2. Extract total_energy
        # 3. Compute e_ads = E_slab_ads - (E_slab + E_ads_mol)
        
        # Mocking for the initial structure
        return {
            "total_energy": -100.5,
            "adsorption_energy": -1.25,
            "surface_energy": 0.05,
            "bader_charges": {}
        }

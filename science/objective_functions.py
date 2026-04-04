import numpy as np
from typing import Dict, Any

class SyntheticSurfaceObjective:
    """
    Provides a ground-truth energy landscape for testing the CLASDE loop.
    Simulates a perovskite surface where specific cation arrangements and 
    vacancy concentrations minimize the energy.
    """
    def __init__(self, target_elements: list = ["Co", "Fe"]):
        self.target_elements = target_elements

    def evaluate(self, state_dict: Dict[str, Any]) -> float:
        """
        Returns a 'total energy' based on composition and vacancies.
        Optimal state: Co=0.2, Fe=0.8, Vacancies=1
        """
        comp = state_dict.get("bulk_composition", {})
        co_frac = comp.get("Co", 0.0)
        fe_frac = comp.get("Fe", 0.0)
        
        # Energy penalty for deviating from optimal Co/Fe ratio
        ratio_penalty = (co_frac - 0.2)**2 + (fe_frac - 0.8)**2
        
        # Vacancy effect: 1 vacancy is optimal, more or less is bad
        defects = state_dict.get("defects", [])
        v_count = sum(1 for d in defects if d.get("type") == "vacancy")
        vacancy_penalty = (v_count - 1)**2
        
        # Base energy + noise
        energy = -15.0 + (ratio_penalty * 10.0) + (vacancy_penalty * 2.0)
        noise = np.random.normal(0, 0.05)
        
        return float(energy + noise)

def calculate_synthetic_reward(energy: float) -> float:
    """Converts energy to a reward (minimize energy -> maximize reward)."""
    return -energy

import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

class ObjectiveFunction(ABC):
    """Base class for all scientific reward functions."""
    @abstractmethod
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        pass

class StabilityObjective(ObjectiveFunction):
    """Minimize total energy (maximize negative energy)."""
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        return -results.get("total_energy", 0.0)

class SabatierObjective(ObjectiveFunction):
    """Target a specific adsorption energy."""
    def __init__(self, target_e_ads: float = -1.5):
        self.target = target_e_ads
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        e_ads = results.get("adsorption_energy", 0.0)
        return -abs(e_ads - self.target)

class UncertaintyObjective(ObjectiveFunction):
    """Maximize model uncertainty (Exploration)."""
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        return results.get("sigma", 1.0)

class ReactionBarrierObjective(ObjectiveFunction):
    """Minimize activation barriers."""
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        return -results.get("activation_barrier", 5.0)

class FunctionalObjective(ObjectiveFunction):
    """Custom expression-based objective."""
    def __init__(self, expression: str):
        self.expression = expression
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        # Placeholder for real expression evaluation
        return results.get("reward", 0.0)

class CompositeObjective(ObjectiveFunction):
    """Weighted sum of multiple objectives."""
    def __init__(self, objectives: Dict[ObjectiveFunction, float]):
        self.objectives = objectives
    def evaluate(self, state: Any, results: Dict[str, Any]) -> float:
        score = 0.0
        for obj, weight in self.objectives.items():
            score += weight * obj.evaluate(state, results)
        return score

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
        
        ratio_penalty = (co_frac - 0.2)**2 + (fe_frac - 0.8)**2
        
        defects = state_dict.get("defects", [])
        v_count = sum(1 for d in defects if d.get("type") == "vacancy")
        vacancy_penalty = (v_count - 1)**2
        
        energy = -15.0 + (ratio_penalty * 10.0) + (vacancy_penalty * 2.0)
        noise = np.random.normal(0, 0.05)
        
        return float(energy + noise)

def calculate_synthetic_reward(energy: float) -> float:
    """Converts energy to a reward (minimize energy -> maximize reward)."""
    return -energy

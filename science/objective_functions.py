import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)

class ObjectiveFunction(ABC):
    """
    Formalized scalar objective: O = f_θ(P(S)).
    Maps physical observables P(S) extracted from simulations into a single scalar reward O.
    
    The Bayesian Optimization surrogate model exclusively trains on this scalar O.
    """
    @abstractmethod
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute the scalar reward from physical observables."""
        pass

# Aliases for backward compatibility or different conceptual layers
RewardFunction = ObjectiveFunction

class StabilityObjective(ObjectiveFunction):
    """Objective based on surface energy minimization: O = -γ_surf."""
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        surface_energy = observables.get("surface_energy")
        if surface_energy is None:
            return -1e9 # Penalty for failed calculation
        return -float(surface_energy)

class SabatierObjective(ObjectiveFunction):
    """
    Sabatier Principle: Activity peaks at an optimal intermediate binding energy.
    Reward = -|E_ads - E_optimum|
    """
    def __init__(self, target_e_ads: float) -> None:
        self.target_e_ads = target_e_ads
        
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        e_ads = observables.get("adsorption_energy")
        if e_ads is None:
            return -1e9
        return -abs(float(e_ads) - self.target_e_ads)

class ReactionBarrierObjective(ObjectiveFunction):
    """
    Minimize the activation barrier of the rate-determining step.
    Reward = -E_a
    """
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        e_a = observables.get("reaction_barrier")
        if e_a is None:
            return -1e9
        return -float(e_a)

class SelectivityObjective(ObjectiveFunction):
    """
    Maximize the energy difference between desired and undesired pathways.
    Reward = E_a(undesired) - E_a(desired)
    """
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        ea_des = observables.get("ea_desired")
        ea_undes = observables.get("ea_undesired")
        if ea_des is None or ea_undes is None:
            return -1e9
        return float(ea_undes - ea_des)

class SegregationObjective(ObjectiveFunction):
    """
    Objective based on surface segregation: O = concentration - 0.1 * surface_energy.
    Focuses on enriching a target species at the surface layer.
    """
    def __init__(self, target_species: str) -> None:
        self.target_species = target_species

    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        counts = observables.get("species_counts", {})
        target_n = counts.get(self.target_species, 0)
        total_n = sum(counts.values()) if counts else 1
        
        if total_n == 0:
            return -1e9
            
        concentration = target_n / total_n
        surface_energy = observables.get("surface_energy", 0.0)
        
        return float(concentration - 0.1 * surface_energy)

class CompositeObjective(ObjectiveFunction):
    """
    Weighted combination of multiple objective functions.
    O = Σ w_i * O_i
    """
    def __init__(self, objectives: Dict[ObjectiveFunction, float]) -> None:
        self.objectives = objectives
        
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        total_reward = 0.0
        for obj_func, weight in self.objectives.items():
            total_reward += weight * obj_func.compute_objective(observables, context)
        return total_reward

class FunctionalObjective(ObjectiveFunction):
    """
    Objective defined as an arbitrary mathematical expression of observables.
    """
    def __init__(self, expression: str) -> None:
        self.expression = expression

    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        try:
            safe_dict = {"abs": abs, "min": min, "max": max, "sqrt": np.sqrt}
            eval_scope = {**safe_dict, **observables, **context}
            return float(eval(self.expression, {"__builtins__": {}}, eval_scope))
        except Exception as e:
            logger.error(f"Error evaluating objective expression '{self.expression}': {e}")
            return -1e9

class UncertaintyObjective(ObjectiveFunction):
    """Objective based on information gain: O = σ_model(S)."""
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        uncertainty = observables.get("uncertainty")
        if uncertainty is None:
            return 0.0
        return float(uncertainty)

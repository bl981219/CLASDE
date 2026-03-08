from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class ObjectiveFunction(ABC):
    """
    Formalized scalar objective: O = f_θ(P(S)).
    Maps physical observables P(S) into a scalar reward for optimization.
    """
    @abstractmethod
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        pass

class SabatierObjective(ObjectiveFunction):
    """
    Sabatier Principle: Activity peaks at an optimal intermediate binding energy.
    Reward = -|E_ads - E_optimum|
    """
    def __init__(self, target_e_ads: float):
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

class SurfaceEnergyObjective(ObjectiveFunction):
    """Minimize surface energy for stability."""
    def compute_objective(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        gamma = observables.get("surface_energy")
        if gamma is None:
            return -1e9
        return -float(gamma)

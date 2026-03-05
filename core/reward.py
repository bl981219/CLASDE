from abc import ABC, abstractmethod
from typing import Dict, Any

class RewardFunction(ABC):
    """
    Formalized reward R = f_θ(P(S)).
    Must be mathematically computable and return a scalar.
    """
    @abstractmethod
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute the scalar reward from observables."""
        pass

class StabilityReward(RewardFunction):
    """Reward based on surface energy minimization: R = -γ_surf."""
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        # Prioritize lower surface energy
        surface_energy = observables.get("surface_energy")
        if surface_energy is None:
            return -1e9 # Penalty for failed calculation
        return -float(surface_energy)

class AdsorptionTuningReward(RewardFunction):
    """Reward based on target adsorption energy: R = -|E_ads - E_optimum|."""
    def __init__(self, target_e_ads: float):
        self.target_e_ads = target_e_ads
        
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        e_ads = observables.get("adsorption_energy")
        if e_ads is None:
            return -1e9
        return -abs(float(e_ads) - self.target_e_ads)

class UncertaintyReward(RewardFunction):
    """Reward based on information gain: R = σ_model(S)."""
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        # This relies on the surrogate model's uncertainty prediction
        uncertainty = observables.get("uncertainty")
        if uncertainty is None:
            return 0.0
        return float(uncertainty)

class DeviationDiscoveryReward(RewardFunction):
    """R = |E_ads_DFT - E_ads_predicted|."""
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        e_ads_dft = observables.get("adsorption_energy")
        e_ads_pred = observables.get("predicted_adsorption_energy")
        if e_ads_dft is None or e_ads_pred is None:
            return -1e9
        return abs(float(e_ads_dft) - float(e_ads_pred))

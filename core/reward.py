from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class RewardFunction(ABC):
    """
    Formalized scalar reward: R = f_θ(P(S)).
    
    This abstract base class ensures that every optimization objective (whether it's 
    simple stability or complex microkinetic TOF) maps the physical observables P(S) 
    extracted from DFT into a single scalar reward R. 
    
    The Bayesian Optimization surrogate model exclusively trains on this scalar R.
    """
    @abstractmethod
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute the scalar reward from physical observables."""
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

class CompositeReward(RewardFunction):
    """
    Weighted combination of multiple reward functions.
    R = Σ w_i * R_i
    """
    def __init__(self, rewards: Dict[RewardFunction, float]):
        self.rewards = rewards
        
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        total_reward = 0.0
        for reward_func, weight in self.rewards.items():
            total_reward += weight * reward_func.compute_reward(observables, context)
        return total_reward

class FunctionalReward(RewardFunction):
    """
    Reward defined as an arbitrary mathematical expression of observables.
    e.g., "-abs(adsorption_energy - (-1.5))"
    """
    def __init__(self, expression: str):
        self.expression = expression

    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        try:
            # Simple eval with observables as local context
            # We add 'abs' to the safe built-ins
            safe_dict = {"abs": abs, "min": min, "max": max, "sqrt": np.sqrt if 'np' in globals() else None}
            # Combine observables and context into the eval scope
            eval_scope = {**safe_dict, **observables, **context}
            return float(eval(self.expression, {"__builtins__": {}}, eval_scope))
        except Exception as e:
            print(f"Error evaluating reward expression '{self.expression}': {e}")
            return -1e9

class SurfacePhaseDiagramReward(RewardFunction):
    """
    Ab initio thermodynamics reward: Ω(T, p) = (E_tot - E_bulk - Σ n_i * μ_i) / A.
    Optimizes for the most stable surface phase under given (T, p) conditions.
    """
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        e_tot = observables.get("total_energy")
        if e_tot is None:
            return -1e9
            
        e_bulk = context.get("e_bulk_ref", 0.0)
        area = context.get("surface_area", 1.0)
        
        # Thermodynamics: ΔG = E_tot - E_bulk - Σ n_i * μ_i(T, p)
        # Simplified: context provides the chemical potential μ_i
        chemical_potentials = context.get("chemical_potentials", {})
        counts = observables.get("species_counts", {}) # n_i
        
        thermo_term = 0.0
        for species, mu in chemical_potentials.items():
            n = counts.get(species, 0)
            thermo_term += n * mu
            
        grand_potential = (e_tot - e_bulk - thermo_term) / area
        return -float(grand_potential) # Minimize grand potential

class ElectrochemicalStabilityReward(RewardFunction):
    """
    Computational Hydrogen Electrode (CHE) reward.
    μ(U) = μ_0 - eU.
    """
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        # Grand potential including electrochemical potential Phi (U)
        phi = context.get("Phi", 0.0)
        # R = - (G_surface(U))
        # Logic similar to phase diagram but with U-dependent chemical potentials
        return self._compute_che_reward(observables, context, phi)

    def _compute_che_reward(self, obs, ctx, phi):
        # Implementation of CHE thermodynamics
        return -1.0 # Placeholder for detailed CHE logic

class MicrokineticReward(RewardFunction):
    """
    Reward based on Turnover Frequency (TOF) or selectivity from microkinetics.
    Bridges the gap between DFT energetics and experimental observables.
    """
    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        # 1. Map DFT energies (E_ads) to intermediate binding energies
        # 2. Build reaction network (e.g., O* + OH* -> H2O*)
        # 3. Solve steady-state microkinetic equations
        # 4. Extract TOF
        
        # Placeholder for microkinetic modeling:
        tof = observables.get("tof", 1e-3)
        return float(np.log10(tof)) # Optimize for log(TOF)

class SegregationReward(RewardFunction):
    """
    Reward based on surface segregation: R = - (G_surface - G_bulk_reference).
    Focuses on enriching a target species at the surface layer.
    """
    def __init__(self, target_species: str):
        self.target_species = target_species

    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        # In this simplified model, reward is proportional to the concentration 
        # of the target species in the topmost layers relative to bulk.
        # R = count_at_surface / total_count
        counts = observables.get("species_counts", {})
        target_n = counts.get(self.target_species, 0)
        total_n = sum(counts.values()) if counts else 1
        
        if total_n == 0:
            return -1e9
            
        concentration = target_n / total_n
        
        # We also want it to be stable, so we might combine it with surface energy
        surface_energy = observables.get("surface_energy", 0.0)
        
        return float(concentration - 0.1 * surface_energy)

class AutonomousDiscoveryReward(RewardFunction):
    """
    Dynamic reward balancing three competing scientific goals:
    R = α * objective_improvement + β * surrogate_uncertainty + γ * novelty
    
    This ensures the engine doesn't just get stuck exploiting one good surface 
    (optimization), but actively explores uncertain regions and pursues novel 
    configurations (scientific discovery).
    """
    def __init__(self, base_reward: RewardFunction, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5):
        self.base_reward = base_reward
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_reward(self, observables: Dict[str, Any], context: Dict[str, Any]) -> float:
        # 1. Exploitation: How good is the physical property?
        objective_reward = self.base_reward.compute_reward(observables, context)
        
        # 2. Exploration: How uncertain was the model? (provided via context)
        uncertainty = context.get("uncertainty", 0.0)
        
        # 3. Novelty: How different is this from previously known states? (provided via context)
        novelty = context.get("novelty", 0.0)
        
        return float(self.alpha * objective_reward + self.beta * uncertainty + self.gamma * novelty)

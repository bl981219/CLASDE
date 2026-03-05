from core.reward import (
    RewardFunction, 
    StabilityReward, 
    AdsorptionTuningReward, 
    UncertaintyReward,
    DeviationDiscoveryReward,
    FunctionalReward,
    CompositeReward
)
from typing import Dict, Any, List, Optional

class ResearchGovernor:
    """
    Agent 1 — Research Governor (The Lab Manager).
    
    This agent enforces the high-level directives of the campaign. It initializes the 
    appropriate `RewardFunction` based on the configuration, tracks the computational 
    budget (e.g., `max_evaluations`), and holds the absolute constraints of the system 
    (e.g., which facets are allowed to be explored).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_function = self._initialize_reward()
        self.max_evaluations = config.get("budget", {}).get("max_evaluations", 50)
        self.current_evaluations = 0

    def _initialize_reward(self) -> RewardFunction:
        obj = self.config.get("objective", {})
        obj_type = obj.get("type", "stability")
        
        if obj_type == "stability":
            return StabilityReward()
        elif obj_type == "adsorption_tuning":
            target = obj.get("target_e_ads", 0.0)
            return AdsorptionTuningReward(target_e_ads=target)
        elif obj_type == "uncertainty_maximization":
            return UncertaintyReward()
        elif obj_type == "deviation_discovery":
            return DeviationDiscoveryReward()
        elif obj_type == "functional":
            expr = obj.get("expression", "0.0")
            return FunctionalReward(expression=expr)
        elif obj_type == "composite":
            components = obj.get("components", [])
            rewards_map = {}
            for comp in components:
                # Recursively initialize component rewards
                temp_gov = ResearchGovernor({"objective": comp})
                weight = comp.get("weight", 1.0)
                rewards_map[temp_gov.get_reward_function()] = weight
            return CompositeReward(rewards=rewards_map)
        else:
            raise ValueError(f"Unsupported objective type: {obj_type}")

    def get_reward_function(self) -> RewardFunction:
        """Provide the current reward function to the loop."""
        return self.reward_function

    def has_budget(self) -> bool:
        """Check if the exploration budget is exhausted."""
        return self.current_evaluations < self.max_evaluations

    def consume_budget(self):
        """Register one evaluation."""
        self.current_evaluations += 1

    def get_constraints(self) -> Dict[str, Any]:
        """Return system constraints (e.g., Miller indices, facets)."""
        return self.config.get("constraints", {})

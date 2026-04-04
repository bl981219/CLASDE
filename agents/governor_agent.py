import logging
from science.objective_functions import (
    ObjectiveFunction, 
    StabilityObjective, 
    SabatierObjective, 
    UncertaintyObjective,
    ReactionBarrierObjective,
    FunctionalObjective,
    CompositeObjective
)
from core.campaign import ResearchMode
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Global Safety Constants
HARD_MAX_BUDGET = 100

class ResearchGovernor:
    """
    The Research Governor Agent (Agent 1).

    In the Perpetual Discovery model, this agent monitors campaign health 
    and determines if scientific goals have been met. It ensures the 
    discovery loop continues 24/7 until manual intervention or 
    convergence is detected.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the governor.
        """
        self.config = config
        mode_val = config.get("research_mode", "tuning")
        self.research_mode = ResearchMode(mode_val)
        self.reward_function = self._initialize_reward()

        # Optional constraints (No default hard budget)
        self.max_evaluations = config.get("budget", {}).get("max_evaluations", float('inf'))
        self.target_reward = config.get("objective", {}).get("target_reward")
        self.min_uncertainty = config.get("objective", {}).get("min_uncertainty", 0.05)
        
        self.current_evaluations = 0

    def should_continue(self, latest_reward: Optional[float] = None, current_uncertainty: float = 1.0) -> bool:
        """
        Determines if the research campaign should continue.
        Defaults to True for investigative research (Perpetual Mode).
        """
        # 1. Enforce Safety Ceiling (Hard Limit)
        if self.current_evaluations >= HARD_MAX_BUDGET:
            logger.warning(f"[Governor] HARD_MAX_BUDGET of {HARD_MAX_BUDGET} reached. Forcing termination.")
            return False

        # 2. Check user-defined budget
        if self.current_evaluations >= self.max_evaluations:
            logger.info("[Governor] User-defined evaluation limit reached.")
            return False
            
        # 2. Check if specific scientific target reached
        if self.target_reward and latest_reward:
            if latest_reward >= self.target_reward:
                logger.info(f"[Governor] Scientific target reached (Reward: {latest_reward:.4f}).")
                return False
                
        # 3. Check for knowledge convergence
        if current_uncertainty < self.min_uncertainty and self.current_evaluations > 10:
            logger.info("[Governor] Model uncertainty below threshold. Knowledge converged.")
            return False

        return True

    def has_budget(self) -> bool:
        """Legacy alias for should_continue()."""
        return self.should_continue()

    def consume_budget(self) -> None:
        """Registers the completion of one iteration."""
        self.current_evaluations += 1

    def get_constraints(self) -> Dict[str, Any]:
        """Returns the physical and chemical constraints of the material space."""
        return self.config.get("constraints", {})

    def _initialize_reward(self) -> ObjectiveFunction:
        """
        Factory method to instantiate the correct ObjectiveFunction based on the goal.
        """
        obj = self.config.get("objective", {})
        obj_type = obj.get("type", "stability")
        
        # Mapping categorical goals to formal mathematical objectives
        if obj_type == "stability":
            return StabilityObjective()
        elif obj_type == "adsorption_tuning":
            target = obj.get("target_e_ads", 0.0)
            return SabatierObjective(target_e_ads=target)
        elif obj_type == "uncertainty_maximization":
            return UncertaintyObjective()
        elif obj_type == "reaction_barrier":
            return ReactionBarrierObjective()
        elif obj_type == "functional":
            expr = obj.get("expression", "0.0")
            return FunctionalObjective(expression=expr)
        elif obj_type == "composite":
            components = obj.get("components", [])
            rewards_map: Dict[ObjectiveFunction, float] = {}
            for comp in components:
                temp_gov = ResearchGovernor({"objective": comp})
                weight = comp.get("weight", 1.0)
                rewards_map[temp_gov.get_reward_function()] = weight
            return CompositeObjective(objectives=rewards_map)
        else:
            raise ValueError(f"Unsupported objective type: {obj_type}")

    def get_reward_function(self) -> ObjectiveFunction:
        """Returns the active reward function for the BO loop."""
        return self.reward_function

    def get_research_mode(self) -> ResearchMode:
        """Returns the active research mode."""
        return self.research_mode

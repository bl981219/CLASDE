from typing import List, Tuple, Dict, Any
import numpy as np
from core.surrogate import SurrogateModel
from core.acquisition import (
    AcquisitionFunction, 
    ExpectedImprovement, 
    UpperConfidenceBound, 
    ThompsonSampling
)
from core.state import SurfaceState
from core.action import MutationAction

class OptimizationStrategist:
    """
    Agent 2 — Optimization Strategist.
    Operates surrogate model, computes acquisition, selects next action.
    Purely algorithmic.
    """
    def __init__(self, surrogate: SurrogateModel, config: Dict[str, Any]):
        self.surrogate = surrogate
        self.config = config

    def select_next_action(
        self, 
        current_state: SurfaceState, 
        candidates: List[Tuple[MutationAction, SurfaceState]], 
        best_f: float
    ) -> Tuple[MutationAction, SurfaceState]:
        """
        Evaluate candidate state transitions and select the best one via acquisition.
        """
        if not candidates:
            raise ValueError("No candidate actions provided to strategist.")

        acq_type = self.config.get("acquisition_type", "EI")
        
        # Instantiate acquisition function
        if acq_type == "EI":
            acq_func = ExpectedImprovement(best_observed_f=best_f)
        elif acq_type == "UCB":
            acq_func = UpperConfidenceBound(kappa=self.config.get("kappa", 2.576))
        elif acq_type == "TS":
            acq_func = ThompsonSampling()
        else:
            raise ValueError(f"Unknown acquisition type: {acq_type}")

        # Score candidates
        scores = []
        for action, state in candidates:
            score = acq_func.compute_score(state, self.surrogate)
            scores.append(score)
            
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    def update_model(self, dataset: List[Dict[str, Any]]):
        """Refit the surrogate model with new data."""
        self.surrogate.update(dataset)

import logging
from typing import List, Dict, Any, Tuple, Optional
from optimization.surrogate_models import SurrogateModel
from optimization.acquisition_functions import AcquisitionFunction
from core.state import SurfaceState
from core.action import MutationAction

logger = logging.getLogger(__name__)

class CampaignOptimizer:
    """
    High-level orchestrator for the Bayesian Optimization process.
    
    This class manages the lifecycle of a surrogate model and the 
    selection of experiments, abstracting the math away from the Agents.
    """
    def __init__(self, surrogate: SurrogateModel, acquisition: AcquisitionFunction) -> None:
        self.surrogate = surrogate
        self.acquisition = acquisition
        
    def update(self, data: List[Dict[str, Any]]) -> None:
        """Refit the surrogate model with new experimental observations."""
        self.surrogate.update(data)
        
    def recommend_next(self, candidates: List[Tuple[MutationAction, SurfaceState]], context: Optional[Dict[str, Any]] = None) -> Tuple[MutationAction, SurfaceState]:
        """
        Evaluate candidates and select the one that maximizes the acquisition function.
        """
        if not candidates:
            logger.error("No candidates provided for recommendation.")
            raise ValueError("No candidates provided for recommendation.")
            
        scores: List[float] = []
        for action, state in candidates:
            # Prepare context for this specific action-state pair
            item_context = (context or {}).copy()
            item_context["action"] = action
            
            score = self.acquisition.compute_score(state, self.surrogate, context=item_context)
            scores.append(score)
            
        import numpy as np
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

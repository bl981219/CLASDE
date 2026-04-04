import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from optimization.surrogate_models import SurrogateModel
from optimization.acquisition_functions import AcquisitionFunction
from core.state import SurfaceState
from core.action import MutationAction
from core.hypothesis import Hypothesis, HypothesisType

logger = logging.getLogger(__name__)

class CampaignOptimizer:
    """
    High-level orchestrator for the Bayesian Optimization process.
    """
    def __init__(self, surrogate: SurrogateModel, acquisition: AcquisitionFunction) -> None:
        self.surrogate = surrogate
        self.acquisition = acquisition
        
    def update(self, data: List[Dict[str, Any]]) -> None:
        """Refit the surrogate model with new experimental observations."""
        self.surrogate.update(data)
        
    def apply_hypothesis(self, candidates: List[Tuple[MutationAction, SurfaceState]], hypothesis: Optional[Hypothesis]) -> List[Tuple[MutationAction, SurfaceState]]:
        """
        Filters or biases candidates based on the active scientific hypothesis.
        """
        if not hypothesis:
            return candidates

        filtered_candidates = []
        
        if hypothesis.type == HypothesisType.CONSTRAINT:
            logger.info(f"Applying constraint hypothesis: {hypothesis.description}")
            constraints = hypothesis.constraints or {}
            for action, state in candidates:
                # Example: Constraint on bulk composition
                if "bulk_composition" in constraints:
                    # Logic to check if state matches constraints
                    pass
                filtered_candidates.append((action, state))
        
        elif hypothesis.type == HypothesisType.PRIOR:
            logger.info(f"Applying prior bias hypothesis: {hypothesis.description}")
            # Priors don't filter, they just stay in the list (biasing happens in scoring)
            filtered_candidates = candidates
            
        else:
            filtered_candidates = candidates

        return filtered_candidates

    def recommend_next(self, candidates: List[Tuple[MutationAction, SurfaceState]], 
                       hypothesis: Optional[Hypothesis] = None,
                       context: Optional[Dict[str, Any]] = None) -> Tuple[MutationAction, SurfaceState]:
        """
        Evaluate candidates and select the one that maximizes the acquisition function,
        considering the active hypothesis.
        """
        if not candidates:
            raise ValueError("No candidates provided.")
            
        # 1. Apply hypothesis-driven filtering
        active_candidates = self.apply_hypothesis(candidates, hypothesis)
        
        if not active_candidates:
            logger.warning("Hypothesis filtered out all candidates. Falling back to original set.")
            active_candidates = candidates

        # 2. Score remaining candidates
        scores: List[float] = []
        for action, state in active_candidates:
            item_context = (context or {}).copy()
            item_context["action"] = action
            
            # 3. Inject hypothesis bias into context for acquisition function
            if hypothesis and hypothesis.type == HypothesisType.PRIOR:
                item_context["hypothesis_prior"] = hypothesis.prior_distribution
            
            score = self.acquisition.compute_score(state, self.surrogate, context=item_context)
            scores.append(score)
            
        best_idx = int(np.argmax(scores))
        return active_candidates[best_idx]

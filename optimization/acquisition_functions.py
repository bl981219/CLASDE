import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from scipy.stats import norm
from .surrogate_models import SurrogateModel
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class AcquisitionFunction(ABC):
    """
    Acquisition function α(S) for selecting the next optimal experiment.
    """
    @abstractmethod
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel, context: Optional[Dict[str, Any]] = None) -> float:
        """Score a candidate state with optional scientific context."""
        pass

class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement (EI) strategy."""
    def __init__(self, best_observed_f: float, xi: float = 0.01) -> None:
        self.best_f = best_observed_f
        self.xi = xi
        
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel, context: Optional[Dict[str, Any]] = None) -> float:
        mu, sigma = surrogate.predict(state)
        if sigma <= 0:
            return 0.0
            
        imp = mu - self.best_f - self.xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return float(ei)

class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound (UCB) strategy: α = μ + κσ."""
    def __init__(self, kappa: float = 2.576) -> None:
        self.kappa = kappa
        
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel, context: Optional[Dict[str, Any]] = None) -> float:
        mu, sigma = surrogate.predict(state)
        return float(mu + self.kappa * sigma)

class ScientificDiscoveryAcquisition(AcquisitionFunction):
    """
    Multi-objective acquisition for true scientific discovery.
    α = Predicted_Reward + β*Uncertainty + γ*Novelty - δ*Cost
    """
    def __init__(self, beta: float = 1.0, gamma: float = 0.5, delta: float = 0.1) -> None:
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel, context: Optional[Dict[str, Any]] = None) -> float:
        mu, sigma = surrogate.predict(state)
        context = context or {}
        
        # 1. Novelty: Distance to existing knowledge
        novelty = 0.0
        existing_feats = context.get("existing_features", [])
        if existing_feats:
            current_feat = np.array(state.feature_vector)
            # Min distance to any existing state
            distances = [np.linalg.norm(current_feat - np.array(f)) for f in existing_feats]
            novelty = float(np.min(distances))
            
        # 2. Cost: Estimated computational resource usage
        # Simple heuristic: VASP is expensive, LOCAL_EMT is cheap
        cost = 1.0
        action = context.get("action")
        if action:
            # We could refine cost based on action type
            cost = 1.0 # Default
            
        return float(mu + self.beta * sigma + self.gamma * novelty - self.delta * cost)

class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling by sampling from the surrogate posterior."""
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel, context: Optional[Dict[str, Any]] = None) -> float:
        mu, sigma = surrogate.predict(state)
        if sigma <= 0:
            return float(mu)
        return float(np.random.normal(mu, sigma))

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from scipy.stats import norm
from .surrogate import SurrogateModel
from .state import SurfaceState

class AcquisitionFunction(ABC):
    """
    Acquisition function α(S) for selecting the next action.
    """
    @abstractmethod
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel) -> float:
        """Score a candidate state."""
        pass

class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement (EI) strategy."""
    def __init__(self, best_observed_f: float, xi: float = 0.01):
        self.best_f = best_observed_f
        self.xi = xi
        
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel) -> float:
        mu, sigma = surrogate.predict(state)
        if sigma <= 0:
            return 0.0
            
        imp = mu - self.best_f - self.xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return float(ei)

class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound (UCB) strategy: α = μ + κσ."""
    def __init__(self, kappa: float = 2.576):
        self.kappa = kappa
        
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel) -> float:
        mu, sigma = surrogate.predict(state)
        return float(mu + self.kappa * sigma)

class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling by sampling from the surrogate posterior."""
    def compute_score(self, state: SurfaceState, surrogate: SurrogateModel) -> float:
        # For simplicity, we sample a single point from the distribution
        mu, sigma = surrogate.predict(state)
        if sigma <= 0:
            return float(mu)
        return float(np.random.normal(mu, sigma))

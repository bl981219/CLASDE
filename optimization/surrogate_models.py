import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Kernel
from sklearn.ensemble import RandomForestRegressor
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class SurrogateModel(ABC):
    """
    A continuously updated regression model f_hat(S) ≈ R.
    
    In Bayesian Optimization, the Surrogate Model serves as a fast, cheap approximation 
    of the true, expensive physical evaluation (DFT or MLFF). It learns a mapping from the 
    `SurfaceState.feature_vector` to the scalar reward `R`.
    
    Crucially, a valid surrogate must output both a prediction (mean) AND an uncertainty 
    (standard deviation). This uncertainty is what drives the exploration of unknown 
    configurations.
    """
    @abstractmethod
    def update(self, dataset: List[Dict[str, Any]]) -> None:
        """Update surrogate training data and refit model."""
        pass

    @abstractmethod
    def predict(self, state: SurfaceState) -> Tuple[float, float]:
        """Predict mean and uncertainty (standard deviation) for a state."""
        pass

class GaussianProcessModel(SurrogateModel):
    """
    Gaussian Process Regression model for surrogate modeling.
    """
    def __init__(self, kernel: Optional[Kernel] = None) -> None:
        if kernel is None:
            # Default kernel for BO: Constant * RBF + White noise
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=25,
            normalize_y=True
        )
        self.is_fitted = False

    def update(self, dataset: List[Dict[str, Any]]) -> None:
        """Update surrogate training data and refit model."""
        X: List[List[float]] = []
        y: List[float] = []
        for entry in dataset:
            state = entry['state']
            if isinstance(state, SurfaceState):
                features = state.feature_vector
            else:
                features = state # Assume precomputed features
            
            X.append(features)
            y.append(entry['reward'])
            
        if len(X) > 0:
            self.model.fit(np.array(X), np.array(y))
            self.is_fitted = True

    def predict(self, state: SurfaceState) -> Tuple[float, float]:
        """Predict mean and uncertainty (standard deviation) for a state."""
        if not self.is_fitted:
            # Prior mean and high uncertainty if not fitted
            return 0.0, 1.0
            
        X = np.array([state.feature_vector])
        mu, sigma = self.model.predict(X, return_std=True)
        return float(mu[0]), float(sigma[0])

class RandomForestModel(SurrogateModel):
    """
    Random Forest Regression model for surrogate modeling.
    Uses forest variance for uncertainty estimation.
    """
    def __init__(self, n_estimators: int = 100, **kwargs: Any) -> None:
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
        self.is_fitted = False

    def update(self, dataset: List[Dict[str, Any]]) -> None:
        """Update surrogate training data and refit model."""
        X: List[List[float]] = []
        y: List[float] = []
        for entry in dataset:
            state = entry['state']
            if isinstance(state, SurfaceState):
                features = state.feature_vector
            else:
                features = state
            
            X.append(features)
            y.append(entry['reward'])
            
        if len(X) > 0:
            self.model.fit(np.array(X), np.array(y))
            self.is_fitted = True

    def predict(self, state: SurfaceState) -> Tuple[float, float]:
        """Predict mean and uncertainty (standard deviation) for a state."""
        if not self.is_fitted:
            return 0.0, 1.0
            
        X = np.array([state.feature_vector])
        
        # Mean prediction
        mu = self.model.predict(X)[0]
        
        # Uncertainty estimation via variance across trees
        preds: List[float] = []
        for tree in self.model.estimators_:
            preds.append(float(tree.predict(X)[0]))
        
        sigma = float(np.std(preds))
        return float(mu), sigma

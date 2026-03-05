from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from .state import SurfaceState

class SurrogateModel:
    """
    A continuously updated regression model f_hat(S) ≈ E_ads.
    Uses Gaussian Process Regression by default.
    """
    def __init__(self, kernel=None):
        if kernel is None:
            # Default kernel for BO: Constant * RBF + White noise
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=25,
            normalize_y=True
        )
        self.is_fitted = False

    def update(self, dataset: List[Dict[str, Any]]):
        """
        Update surrogate training data and refit model.
        dataset: List of dicts containing 'state' and 'target_value'
        """
        X = []
        y = []
        for entry in dataset:
            state = entry['state']
            if isinstance(state, SurfaceState):
                features = state.feature_vector
            else:
                features = state # Assume precomputed features
            
            X.append(features)
            y.append(entry['target_value'])
            
        if len(X) > 0:
            self.model.fit(np.array(X), np.array(y))
            self.is_fitted = True

    def predict(self, state: SurfaceState) -> Tuple[float, float]:
        """
        Predict mean and uncertainty (standard deviation) for a state.
        returns: (mean, sigma)
        """
        if not self.is_fitted:
            # Prior mean and high uncertainty if not fitted
            return 0.0, 1.0
            
        X = np.array([state.feature_vector])
        mu, sigma = self.model.predict(X, return_std=True)
        return float(mu[0]), float(sigma[0])

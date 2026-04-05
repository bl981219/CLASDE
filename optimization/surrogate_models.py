import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Kernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class SurrogateModel(ABC):
    """
    A continuously updated regression model f_hat(S) ≈ R.
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
    Gaussian Process Regression model with ARD kernel and Dimensionality Reduction.
    """
    def __init__(self, kernel: Optional[Kernel] = None, use_pca: bool = True, n_components: int = 16) -> None:
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.pca_active = False # Tracks if PCA was actually used in the last fit
        
        self.kernel_template = kernel
        self.model = None
        self.is_fitted = False

    def _init_model(self, n_features: int):
        if self.kernel_template is not None:
            kernel = self.kernel_template
        else:
            ls = np.ones(n_features)
            ls_bounds = (1e-2, 1e3)
            kernel = C(1.0, (1e-3, 1e3)) * RBF(ls, ls_bounds) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5,
            normalize_y=True
        )

    def update(self, dataset: List[Dict[str, Any]]) -> None:
        """Update surrogate training data and refit model."""
        X_raw: List[List[float]] = []
        y: List[float] = []
        for entry in dataset:
            state = entry['state']
            features = state.get_feature_vector() if isinstance(state, SurfaceState) else state
            X_raw.append(features)
            y.append(entry['reward'])
            
        if len(X_raw) == 0:
            return

        X = np.array(X_raw)
        self.pca_active = False
        if self.use_pca and X.shape[0] >= 2:
            n_comp = min(self.n_components, X.shape[0])
            if self.pca.n_components != n_comp:
                self.pca = PCA(n_components=n_comp)
            X = self.pca.fit_transform(X)
            self.pca_active = True

        if self.model is None or self.model.kernel.n_dims != X.shape[1]:
            # Re-init model if dimension changed (e.g. PCA components increased)
            self._init_model(X.shape[1])
            
        self.model.fit(X, np.array(y))
        self.is_fitted = True

    def predict(self, state: SurfaceState) -> Tuple[float, float]:
        """Predict mean and uncertainty (standard deviation) for a state."""
        if not self.is_fitted or self.model is None:
            return 0.0, 1.0
            
        X = np.array([state.get_feature_vector()])
        if self.pca_active:
            X = self.pca.transform(X)
            
        mu, sigma = self.model.predict(X, return_std=True)
        return float(mu[0]), float(sigma[0])

class RandomForestModel(SurrogateModel):
    """
    Random Forest Regression model for surrogate modeling.
    """
    def __init__(self, n_estimators: int = 100, **kwargs: Any) -> None:
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
        self.is_fitted = False

    def update(self, dataset: List[Dict[str, Any]]) -> None:
        X: List[List[float]] = []
        y: List[float] = []
        for entry in dataset:
            state = entry['state']
            features = state.get_feature_vector() if isinstance(state, SurfaceState) else state
            X.append(features)
            y.append(entry['reward'])
            
        if len(X) > 0:
            self.model.fit(np.array(X), np.array(y))
            self.is_fitted = True

    def predict(self, state: SurfaceState) -> Tuple[float, float]:
        if not self.is_fitted:
            return 0.0, 1.0
            
        X = np.array([state.get_feature_vector()])
        mu = self.model.predict(X)[0]
        preds = [float(tree.predict(X)[0]) for tree in self.model.estimators_]
        sigma = float(np.std(preds))
        return float(mu), sigma

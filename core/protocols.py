"""
Protocol definitions for CLASDE interfaces.

Protocols provide structural subtyping (duck typing with type safety)
for key abstractions in the system.
"""

from typing import Protocol, Dict, Any, List, Optional, Tuple, runtime_checkable
import numpy as np
from core.state import SurfaceState
from core.action import MutationAction


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for any LLM backend (Gemini, OpenAI, Claude, etc.)."""
    
    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate structured output from LLM.
        
        Args:
            prompt: User prompt
            system_instruction: System/role instruction
            response_format: Expected format ("json" or "text")
            
        Returns:
            Parsed response as dictionary
        """
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Interface for experiment storage backends."""
    
    def add_experiment(
        self,
        state: SurfaceState,
        results: Dict[str, Any],
        action: Optional[MutationAction] = None,
        parent_state: Optional[SurfaceState] = None
    ) -> None:
        """Store experiment result with provenance."""
        ...
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Retrieve all experiments for surrogate training."""
        ...
    
    def get_by_id(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single experiment by ID."""
        ...
    
    def get_best_reward(self) -> float:
        """Get best reward across all experiments."""
        ...


@runtime_checkable
class SurrogateModel(Protocol):
    """Interface for surrogate models (GP, NN, etc.)."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model on data."""
        ...
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.
        
        Returns:
            (mean predictions, std predictions)
        """
        ...
    
    def acquisition(self, X: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """Compute acquisition function values."""
        ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Interface for compute backends (Slurm, local, cloud)."""
    
    def submit_job(self, job_spec: Any) -> str:
        """
        Submit computational job.
        
        Returns:
            Job ID for monitoring
        """
        ...
    
    def monitor_job(self, job_id: str) -> str:
        """
        Check job status.
        
        Returns:
            Job status string
        """
        ...
    
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Fetch results from completed job."""
        ...


@runtime_checkable
class ObjectiveFunction(Protocol):
    """Interface for objective/reward functions."""
    
    def evaluate(self, state: SurfaceState, observables: Dict[str, Any]) -> float:
        """
        Compute reward for a given state.
        
        Args:
            state: Surface configuration
            observables: Computed properties (energies, etc.)
            
        Returns:
            Scalar reward value
        """
        ...


@runtime_checkable
class AcquisitionFunction(Protocol):
    """Interface for acquisition strategies."""
    
    def score(
        self,
        candidates: List[SurfaceState],
        surrogate: SurrogateModel,
        memory: MemoryStore
    ) -> np.ndarray:
        """
        Score candidate states for exploration.
        
        Returns:
            Acquisition scores for each candidate
        """
        ...


@runtime_checkable
class EventBus(Protocol):
    """Interface for event publishing/subscription."""
    
    def publish(self, event: Any) -> None:
        """Publish event to subscribers."""
        ...
    
    def subscribe(self, event_type: str, callback: Any) -> None:
        """Register event handler."""
        ...


@runtime_checkable
class CheckpointManager(Protocol):
    """Interface for campaign state persistence."""
    
    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Save campaign state to disk."""
        ...
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load campaign state from disk."""
        ...
    
    def checkpoint_exists(self) -> bool:
        """Check if checkpoint file exists."""
        ...


@runtime_checkable
class TraceLogger(Protocol):
    """Interface for knowledge trace logging."""
    
    def log_trace(self, trace: Any) -> None:
        """Append knowledge trace to log."""
        ...
    
    def get_traces(self) -> List[Any]:
        """Retrieve all traces."""
        ...

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all true autonomous agents in CLASDE.
    
    A true agent is not just a procedural wrapper; it maintains a belief state,
    reasons about uncertainty, and operates via a continuous feedback loop:
    observe -> update_belief -> propose -> score -> execute -> update_memory.
    """
    def __init__(self) -> None:
        self.belief_state: Optional[Any] = None

    @abstractmethod
    def observe_state(self) -> Any:
        """Observe the current environment or memory."""
        pass

    @abstractmethod
    def update_belief(self, observations: Any) -> None:
        """Update internal belief state (e.g., surrogate models, hypotheses) based on new observations."""
        pass

    @abstractmethod
    def propose_actions(self) -> List[Any]:
        """Generate candidate actions based on the current belief state."""
        pass

    @abstractmethod
    def score_actions(self, actions: List[Any]) -> List[float]:
        """Evaluate actions, explicitly weighing expected reward vs. uncertainty."""
        pass

    @abstractmethod
    def execute_best(self, best_action: Any) -> Any:
        """Commit to an action and interface with execution layers (compute/builder)."""
        pass

    @abstractmethod
    def update_memory(self, result: Any) -> None:
        """Commit the results of the execution to the central Knowledge/Memory Graphs."""
        pass

    def run_step(self) -> Any:
        """Executes a single, full cycle of the agentic loop."""
        obs = self.observe_state()
        self.update_belief(obs)
        candidates = self.propose_actions()
        scores = self.score_actions(candidates)
        
        # Select best
        import numpy as np
        best_idx = int(np.argmax(scores))
        best_action = candidates[best_idx]
        
        result = self.execute_best(best_action)
        self.update_memory(result)
        return result

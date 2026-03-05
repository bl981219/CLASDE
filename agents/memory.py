from typing import List, Dict, Any, Tuple
import json
import networkx as nx
from pydantic import BaseModel, Field
from core.state import SurfaceState
from core.action import MutationAction

class MemoryGraph:
    """
    Maintains the exploration history and dataset D = {S, P(S), R}.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.dataset: List[Dict[str, Any]] = []

    def add_state(self, state: SurfaceState, observables: Dict[str, Any] = None, reward: float = None):
        """Add a state and its evaluated properties to memory."""
        state_id = state.get_id()
        if not self.graph.has_node(state_id):
            self.graph.add_node(state_id, state=state, observables=observables, reward=reward)
        else:
            if observables is not None:
                self.graph.nodes[state_id]['observables'] = observables
            if reward is not None:
                self.graph.nodes[state_id]['reward'] = reward
        
        if reward is not None:
            self.dataset.append({
                'state': state,
                'target_value': reward,
                'observables': observables
            })

    def add_transition(self, source: SurfaceState, action: MutationAction, target: SurfaceState):
        """Record a state transition in the graph."""
        self.graph.add_edge(source.get_id(), target.get_id(), action=action)

    def get_best_reward(self) -> float:
        """Return the maximum reward observed so far."""
        rewards = [d['target_value'] for d in self.dataset if d['target_value'] is not None]
        return max(rewards) if rewards else -1e9

    def get_training_data(self) -> List[Dict[str, Any]]:
        """Retrieve data for surrogate model training."""
        return self.dataset

    def save(self, file_path: str):
        """Serialize memory to file (simplified)."""
        # Note: networkx graphs with pydantic objects need custom serialization
        pass

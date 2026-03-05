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
            self.graph.nodes[state_id]['state'] = state
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
        """Serialize memory to a JSON file."""
        data = {
            "dataset": [
                {
                    "state": d["state"].model_dump(),
                    "target_value": d["target_value"],
                    "observables": d["observables"]
                } for d in self.dataset
            ],
            "graph": {
                "nodes": [],
                "edges": []
            }
        }
        
        # Serialize Nodes
        for node_id, node_data in self.graph.nodes(data=True):
            data["graph"]["nodes"].append({
                "id": node_id,
                "state": node_data["state"].model_dump(),
                "observables": node_data.get("observables"),
                "reward": node_data.get("reward")
            })
            
        # Serialize Edges
        for u, v, edge_data in self.graph.edges(data=True):
            data["graph"]["edges"].append({
                "source": u,
                "target": v,
                "action": edge_data["action"].model_dump()
            })
            
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: str):
        """Load memory from a JSON file."""
        import os
        if not os.path.exists(file_path):
            return
            
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Clear existing
        self.graph = nx.DiGraph()
        self.dataset = []
        
        # Nodes must be added first
        for n in data.get("graph", {}).get("nodes", []):
            state = SurfaceState(**n["state"])
            self.add_state(state, n.get("observables"), n.get("reward"))
            
        # Then edges
        for e in data.get("graph", {}).get("edges", []):
            u_data = next(n for n in data["graph"]["nodes"] if n["id"] == e["source"])
            v_data = next(n for n in data["graph"]["nodes"] if n["id"] == e["target"])
            source = SurfaceState(**u_data["state"])
            target = SurfaceState(**v_data["state"])
            action = MutationAction(**e["action"])
            self.add_transition(source, action, target)

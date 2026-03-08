import logging
from typing import List, Dict, Any, Tuple, Optional
import json
import os
import networkx as nx
from core.state import SurfaceState
from core.action import MutationAction

logger = logging.getLogger(__name__)

class ExperimentDatabase:
    """
    Centralized repository for all atomistic experiments.
    
    Stores detailed physical and computational metadata:
    - structure: Atomic configurations (SurfaceState)
    - adsorption energy: Calculated energetic properties
    - method: DFT (VASP/PBE), MLFF (MACE/EMT)
    - convergence: SCF/Ionic steps, force thresholds
    """
    def __init__(self, storage_path: str = "data/results/experiment_db.json") -> None:
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        self.dataset: List[Dict[str, Any]] = []

    def add_experiment(self, state: SurfaceState, results: Dict[str, Any], 
                       action: Optional[MutationAction] = None, 
                       parent_state: Optional[SurfaceState] = None) -> None:
        """Add a complete experiment record to the database."""
        state_id = state.get_id()
        
        # 1. Store in flat dataset for BO training
        record = {
            "state": state,
            "reward": results.get("reward"),
            "observables": results,
            "method": results.get("fidelity", "unknown"),
            "convergence": results.get("convergence", True)
        }
        self.dataset.append(record)
        
        # 2. Store in provenance graph
        self.graph.add_node(state_id, **record)
        if parent_state and action:
            self.graph.add_edge(parent_state.get_id(), state_id, action=action)

    def get_training_data(self) -> List[Dict[str, Any]]:
        return self.dataset

    def get_best_reward(self) -> float:
        rewards = [d['reward'] for d in self.dataset if d['reward'] is not None]
        return float(max(rewards)) if rewards else -1e9

    def save(self) -> None:
        """Serialize DB to disk."""
        data = {
            "experiments": [
                {
                    "state": d["state"].model_dump(),
                    "reward": d["reward"],
                    "observables": d["observables"],
                    "method": d.get("method"),
                    "convergence": d.get("convergence")
                } for d in self.dataset
            ],
            "provenance": []
        }
        for u, v, attr in self.graph.edges(data=True):
            data["provenance"].append({
                "source": u, "target": v, 
                "action": attr["action"].model_dump() if attr.get("action") else None
            })
            
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load DB from disk."""
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, "r") as f:
            data = json.load(f)
        
        self.dataset = []
        self.graph = nx.DiGraph()
        
        for e in data.get("experiments", []):
            state = SurfaceState(**e["state"])
            res = {**e["observables"], "reward": e["reward"], "fidelity": e["method"], "convergence": e["convergence"]}
            # We don't have parent info here easily, so we add nodes first
            self.add_experiment(state, res)
            
        for p in data.get("provenance", []):
            if p["action"]:
                action = MutationAction(**p["action"])
                self.graph.add_edge(p["source"], p["target"], action=action)

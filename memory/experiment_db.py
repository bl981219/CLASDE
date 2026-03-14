import logging
import sqlite3
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from core.state import SurfaceState
from core.action import MutationAction

logger = logging.getLogger(__name__)

class ExperimentDatabase:
    """
    SQL-backed repository for all atomistic experiments.
    Ensures data integrity and enables complex scientific queries.
    """
    def __init__(self, db_path: str = "data/results/experiments.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    state_id TEXT PRIMARY KEY,
                    state_json TEXT,
                    reward REAL,
                    fidelity TEXT,
                    observables_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance (
                    parent_id TEXT,
                    child_id TEXT,
                    action_json TEXT,
                    FOREIGN KEY(parent_id) REFERENCES experiments(state_id),
                    FOREIGN KEY(child_id) REFERENCES experiments(state_id)
                )
            """)

    def add_experiment(self, state: SurfaceState, results: Dict[str, Any], 
                       action: Optional[MutationAction] = None, 
                       parent_state: Optional[SurfaceState] = None) -> None:
        """Saves a complete experiment record to SQL."""
        state_id = state.get_id()
        # Clean observables for JSON storage
        clean_obs = {k: v for k, v in results.items() if k != "structure"}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments 
                (state_id, state_json, reward, fidelity, observables_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                state_id, 
                state.model_dump_json(exclude={'slab_structure'}),
                results.get("reward"),
                results.get("fidelity", "unknown"),
                json.dumps(clean_obs)
            ))
            
            if parent_state and action:
                conn.execute("""
                    INSERT INTO provenance (parent_id, child_id, action_json)
                    VALUES (?, ?, ?)
                """, (parent_state.get_id(), state_id, action.model_dump_json()))

    def get_training_data(self) -> List[Dict[str, Any]]:
        """Retrieves all experiments for surrogate model training with full state hydration."""
        dataset = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM experiments").fetchall()
            for row in rows:
                state = SurfaceState(**json.loads(row["state_json"]))
                dataset.append({
                    "state": state,
                    "reward": row["reward"],
                    "fidelity": row["fidelity"],
                    "observables": json.loads(row["observables_json"])
                })
        return dataset

    def get_best_reward(self) -> float:
        """Quickly query the current best reward across all experiments."""
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT MAX(reward) FROM experiments").fetchone()
            return float(res[0]) if res and res[0] is not None else -1e9

    def save(self):
        """No-op for SQL as it is persistent by default."""
        pass

    def load(self):
        """No-op for SQL."""
        pass

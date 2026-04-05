"""
Result processing service.

Applies experiment results to system state, experiment DB, vector index, and
governor budget consumption.
"""

from typing import Any, Dict

import numpy as np


class ResultProcessor:
    """Processes a single experiment result into memory and campaign state."""

    def __init__(self, *, storage: Any, governor: Any, system_state: Any) -> None:
        self.storage = storage
        self.governor = governor
        self.system_state = system_state

    def process(self, result: Dict[str, Any]) -> None:
        reward = result.get("reward", -1e9)
        if reward > self.system_state.current_best_reward:
            self.system_state.current_best_reward = reward

        state = result["state"]
        self.storage.experiment_db.add_experiment(
            state=state,
            results={**result["observables"], "reward": reward},
            action=result.get("action"),
        )

        feat = np.array(state.get_feature_vector())
        self.storage.vector_index.add_item(feat, {"state_id": state.get_id()})

        self.governor.consume_budget()


"""
Baseline initialization service.

Bootstraps campaign memory with an initial baseline calculation when no prior
training data is available.
"""

import json
import logging
import os
from typing import Any, Dict

from core.state import SurfaceState
from execution.compute_agent import JobStatus, SimulationType

logger = logging.getLogger(__name__)


class BaselineInitializer:
    """Runs initial baseline calculation and stores its result."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        storage: Any,
        builder: Any,
        compute_manager: Any,
        result_processor: Any,
    ) -> None:
        self.config = config
        self.storage = storage
        self.builder = builder
        self.compute_manager = compute_manager
        self.result_processor = result_processor

    def initialize_if_needed(self) -> None:
        """Initialize baseline only when memory has no training data."""
        if self.storage.experiment_db.get_training_data():
            return

        logger.info("[Technician] No data found. Running initial baseline...")

        bulk = self.config.get("constraints", {}).get("bulk", {"Cu": 1.0})
        facet = self.config.get("constraints", {}).get("facet", [1, 1, 1])

        initial_state = SurfaceState(
            bulk_composition=bulk,
            miller_index=tuple(facet),
            termination="default",
        )

        slab = self.builder.build_structure(initial_state)
        initial_state.slab_structure = slab

        job_id = self.compute_manager.submit_job(
            slab, initial_state, SimulationType.MLIP, iteration=0
        )

        job_status = self.compute_manager.poll_for_completion(job_id)
        if job_status != JobStatus.COMPLETED:
            raise RuntimeError(
                f"Baseline job {job_id} ended with status {job_status}. "
                "Check compute environment."
            )

        calc_dir = self.compute_manager.fetch_results(job_id)
        res_path = os.path.join(calc_dir, "results.json")
        if not os.path.exists(res_path):
            raise RuntimeError(
                f"Baseline job {job_id} completed but results.json not found in {calc_dir}."
            )

        with open(res_path, "r") as f:
            raw_data = json.load(f)

        result = {
            "state": initial_state,
            "action": None,
            "reward": raw_data.get("reward", 0.0),
            "observables": raw_data,
            "metadata": {"job_id": job_id, "fidelity": "baseline"},
        }
        self.result_processor.process(result)


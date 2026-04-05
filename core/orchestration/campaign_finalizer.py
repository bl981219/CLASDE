"""
Campaign finalization service.

Runs end-of-campaign theory synthesis, report generation, and persistence.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class CampaignFinalizer:
    """Finalizes campaign artifacts and report."""

    def __init__(self, *, theory_builder: Any, storage: Any, results_dir: str) -> None:
        self.theory_builder = theory_builder
        self.storage = storage
        self.results_dir = results_dir

    def finalize(self) -> None:
        logger.info("[LAB] Discovery complete. Generating Report.")

        self.theory_builder.identify_electronic_descriptors()
        self.theory_builder.identify_termination_bias()

        report = self.theory_builder.generate_report()
        report_path = os.path.join(self.results_dir, "discovery_report.md")
        with open(report_path, "w") as f:
            f.write(report)

        self.storage.save_all()
        logger.info("[LAB] Discovery Report saved to %s", report_path)


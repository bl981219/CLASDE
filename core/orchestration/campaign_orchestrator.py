"""
Campaign orchestration engine.

Extracts the execution loop from CampaignManager so orchestration logic
is isolated from setup, persistence, and reporting responsibilities.
"""

import logging
import time
from typing import Any, Callable, Dict

from core.lab_objects import KnowledgeTrace

logger = logging.getLogger(__name__)


class CampaignOrchestrator:
    """Executes the hierarchical campaign loop with injected collaborators."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        governor: Any,
        pi: Any,
        postdoc: Any,
        technician: Any,
        storage: Any,
        system_state: Any,
        on_initialize_baseline: Callable[[], None],
        on_log_trace: Callable[[KnowledgeTrace], None],
        on_process_result: Callable[[Dict[str, Any]], None],
        on_finalize: Callable[[], None],
    ) -> None:
        self.config = config
        self.governor = governor
        self.pi = pi
        self.postdoc = postdoc
        self.technician = technician
        self.storage = storage
        self.system_state = system_state
        self.on_initialize_baseline = on_initialize_baseline
        self.on_log_trace = on_log_trace
        self.on_process_result = on_process_result
        self.on_finalize = on_finalize

    def run(self) -> None:
        """Execute the strict hierarchical discovery loop."""
        logger.info("--- [CLASDE V5] Starting Hierarchical Discovery Loop ---")

        try:
            self.on_initialize_baseline()
        except Exception as e:
            logger.error("[LAB] Baseline initialization failed: %s", e, exc_info=True)
            raise RuntimeError("Cannot start campaign: baseline initialization failed.") from e

        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.governor.should_continue(
            latest_reward=self.system_state.current_best_reward
        ):
            try:
                idea = self.pi.propose_idea(
                    self.config.get("original_prompt", "Optimize surface stability")
                )

                max_revisions = 3
                revision_count = 0
                critique = self.postdoc.review_idea(idea)

                while not critique.validity and revision_count < max_revisions:
                    logger.warning(
                        "[LAB] PI idea rejected. Revision %s/%s",
                        revision_count + 1,
                        max_revisions,
                    )
                    idea = self.pi.refine_idea(
                        idea,
                        critique.revised_plan or "Provide more physical justification.",
                    )
                    critique = self.postdoc.review_idea(idea)
                    revision_count += 1

                if not critique.validity:
                    logger.error(
                        "[LAB] Failed to converge on a valid idea after max revisions. "
                        "Terminating loop."
                    )
                    break

                hypothesis = self.postdoc.formulate_hypothesis(idea, critique)

                training_data = self.storage.experiment_db.get_training_data()
                if not training_data:
                    logger.error("[LAB] No training data available for experiment design.")
                    break

                current_state = training_data[-1]["state"]
                experiments = self.postdoc.design_experiments(hypothesis, current_state)

                results = self.technician.run_experiments(
                    experiments, self.system_state.iteration
                )

                insight = self.postdoc.analyze_results(hypothesis, results)
                logger.info("[LAB] Insight: %s", insight.conclusion)

                trace = KnowledgeTrace(
                    iteration=self.system_state.iteration,
                    input_idea=idea,
                    critique=critique,
                    final_hypothesis=hypothesis,
                    experiments=experiments,
                    insight=insight,
                )
                self.on_log_trace(trace)

                for res in results:
                    self.on_process_result(res)

                self.system_state.iteration += 1
                self.storage.save_all()
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    "[LAB] Error in knowledge loop (Error %s/%s): %s",
                    consecutive_errors,
                    max_consecutive_errors,
                    e,
                    exc_info=True,
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "[LAB] Too many consecutive errors. Terminating loop for safety."
                    )
                    raise RuntimeError(
                        f"Campaign terminated after {max_consecutive_errors} "
                        "consecutive failures."
                    ) from e

                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                wait = min(5 * (2 ** (consecutive_errors - 1)), 60)
                logger.info("[LAB] Retrying in %ds...", wait)
                time.sleep(wait)
                continue

        self.on_finalize()

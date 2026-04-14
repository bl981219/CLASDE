from unittest.mock import MagicMock, call, patch

import pytest

from core.lab_objects import (
    Critique, Experiment, Hypothesis, Insight, ResearchIdea,
)
from core.orchestration.campaign_orchestrator import CampaignOrchestrator


def _make_orchestrator(config, surface_state, *, governor_side_effect=None):
    """Build a CampaignOrchestrator with fully-mocked dependencies.

    Returns (orchestrator, pi, postdoc, technician, governor, callbacks).
    """
    idea = ResearchIdea(goal="G", intuition="I")

    governor = MagicMock()
    governor.should_continue.side_effect = (
        governor_side_effect if governor_side_effect is not None else [True, False]
    )

    pi = MagicMock()
    pi.propose_idea.return_value = idea
    pi.refine_idea.return_value = idea

    postdoc = MagicMock()
    postdoc.review_idea.return_value = Critique(
        idea_id=idea.id, validity=True, issues=[], confidence=0.9
    )
    postdoc.formulate_hypothesis.return_value = Hypothesis(
        idea_id=idea.id,
        variable="v",
        manipulation="m",
        expected_effect="e",
        metric="stability",
        falsification_condition="f",
    )
    postdoc.design_experiments.return_value = [
        Experiment(hypothesis_id="h1", parameters={}, method="MLIP", expected_output=[])
    ]
    postdoc.analyze_results.return_value = Insight(
        hypothesis_id="h1", conclusion="Done", confidence=0.9
    )

    technician = MagicMock()
    technician.run_experiments.return_value = [
        {
            "reward": -1.0,
            "state": surface_state,
            "action": MagicMock(),
            "observables": {},
            "metadata": {},
            "status": "completed",
        }
    ]

    storage = MagicMock()
    storage.experiment_db.get_training_data.return_value = [{"state": surface_state}]

    system_state = MagicMock()
    system_state.current_best_reward = -1e9
    system_state.iteration = 0

    callbacks = {
        "on_initialize_baseline": MagicMock(),
        "on_log_trace": MagicMock(),
        "on_process_result": MagicMock(),
        "on_finalize": MagicMock(),
    }

    orchestrator = CampaignOrchestrator(
        config=config,
        governor=governor,
        pi=pi,
        postdoc=postdoc,
        technician=technician,
        storage=storage,
        system_state=system_state,
        **callbacks,
    )
    return orchestrator, pi, postdoc, technician, governor, callbacks


def test_single_iteration_calls_all_stages(mock_config, mock_surface_state):
    """All pipeline stages are invoked exactly once during a clean single iteration."""
    orchestrator, pi, postdoc, technician, _, callbacks = _make_orchestrator(
        mock_config, mock_surface_state
    )
    orchestrator.run()

    pi.propose_idea.assert_called_once()
    postdoc.review_idea.assert_called_once()
    postdoc.formulate_hypothesis.assert_called_once()
    postdoc.design_experiments.assert_called_once()
    technician.run_experiments.assert_called_once()
    postdoc.analyze_results.assert_called_once()
    callbacks["on_log_trace"].assert_called_once()
    callbacks["on_finalize"].assert_called_once()


def test_baseline_failure_raises_with_clear_message(mock_config, mock_surface_state):
    """RuntimeError from baseline init is re-raised with a message naming the cause."""
    orchestrator, *_, callbacks = _make_orchestrator(mock_config, mock_surface_state)
    callbacks["on_initialize_baseline"].side_effect = RuntimeError("compute down")

    with pytest.raises(RuntimeError, match="baseline initialization failed"):
        orchestrator.run()


def test_consecutive_errors_terminate_loop(mock_config, mock_surface_state):
    """Loop raises RuntimeError after exceeding the max consecutive error threshold."""
    orchestrator, _, _, technician, governor, _ = _make_orchestrator(
        mock_config, mock_surface_state, governor_side_effect=None
    )
    governor.should_continue.side_effect = None
    governor.should_continue.return_value = True  # never stop on its own
    technician.run_experiments.side_effect = ValueError("job failed")

    with pytest.raises(RuntimeError, match="consecutive failures"):
        orchestrator.run()


def test_exponential_backoff_increases_wait(mock_config, mock_surface_state):
    """time.sleep wait doubles between consecutive errors before eventual success."""
    call_count = 0

    def _run_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ValueError("transient error")
        return [
            {
                "reward": -1.0,
                "state": mock_surface_state,
                "action": MagicMock(),
                "observables": {},
                "metadata": {},
                "status": "completed",
            }
        ]

    # governor: fail, fail, succeed, stop
    orchestrator, _, _, technician, governor, _ = _make_orchestrator(
        mock_config, mock_surface_state, governor_side_effect=[True, True, True, False]
    )
    technician.run_experiments.side_effect = _run_side_effect

    with patch("core.orchestration.campaign_orchestrator.time.sleep") as mock_sleep:
        orchestrator.run()

    sleep_calls = mock_sleep.call_args_list
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == call(5)   # min(5 * 2^0, 60)
    assert sleep_calls[1] == call(10)  # min(5 * 2^1, 60)

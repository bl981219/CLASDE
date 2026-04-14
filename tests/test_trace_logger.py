import json
import pytest
from core.lab_objects import (
    Critique, Experiment, Hypothesis, Insight, KnowledgeTrace, ResearchIdea,
)
from core.orchestration.trace_logger import TraceLogger


def _make_trace(iteration: int = 0) -> KnowledgeTrace:
    idea = ResearchIdea(goal="test goal", intuition="test intuition")
    critique = Critique(idea_id=idea.id, validity=True, issues=[], confidence=0.9)
    hypothesis = Hypothesis(
        idea_id=idea.id,
        variable="dopant",
        manipulation="increase x",
        expected_effect="lower adsorption energy",
        metric="E_ads",
        falsification_condition="no energy change",
    )
    experiment = Experiment(
        hypothesis_id=hypothesis.id,
        parameters={},
        method="MLIP",
        expected_output=["energy"],
    )
    insight = Insight(
        hypothesis_id=hypothesis.id,
        conclusion="Test conclusion",
        confidence=0.8,
    )
    return KnowledgeTrace(
        iteration=iteration,
        input_idea=idea,
        critique=critique,
        final_hypothesis=hypothesis,
        experiments=[experiment],
        insight=insight,
    )


def test_count_is_zero_without_file(tmp_path):
    tl = TraceLogger(str(tmp_path / "traces.jsonl"))
    assert tl.count_traces() == 0


def test_log_and_count(tmp_path):
    tl = TraceLogger(str(tmp_path / "traces.jsonl"))
    for i in range(3):
        tl.log_trace(_make_trace(i))
    assert tl.count_traces() == 3


def test_get_trace_by_iteration(tmp_path):
    tl = TraceLogger(str(tmp_path / "traces.jsonl"))
    for i in range(3):
        tl.log_trace(_make_trace(i))
    result = tl.get_trace_by_iteration(1)
    assert result is not None
    assert result.iteration == 1


def test_export_to_json_creates_valid_file(tmp_path):
    tl = TraceLogger(str(tmp_path / "traces.jsonl"))
    for i in range(2):
        tl.log_trace(_make_trace(i))
    export_path = str(tmp_path / "export.json")
    tl.export_traces_to_json(export_path)
    with open(export_path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["iteration"] == 0
    assert data[1]["iteration"] == 1

import pytest
from unittest.mock import MagicMock, patch
from core.campaign_manager import CampaignManager
from core.state import SurfaceState
from core.action import MutationAction, ActionType
from core.workflow_graph import WorkflowGraph, TaskNode, TaskType
import os

@pytest.fixture
def mock_config():
    return {
        "name": "Test_Campaign",
        "objective": {"type": "stability"},
        "constraints": {"bulk": {"Cu": 1.0}, "facet": [1, 1, 1]},
        "optimization": {"batch_size": 2},
        "compute": {"platform": "local", "mode": "mock"},
        "budget": {"max_evaluations": 2}
    }

@patch('core.campaign_manager.LLMCollaborator')
@patch('core.campaign_manager.ComputeManager')
def test_campaign_manager_init(mock_compute, mock_collab, mock_config):
    manager = CampaignManager(mock_config)
    assert manager.batch_size == 2
    assert manager.storage.is_loaded is True

@patch('core.campaign_manager.ComputeManager')
@patch('core.campaign_manager.HypothesisAgent')
def test_dynamic_planning_logic(mock_pi, mock_compute, mock_config):
    # Mock hypothesis that should trigger MD
    mock_hyp = MagicMock()
    mock_hyp.theory_statement = "The surface is unstable."
    mock_pi.return_value.formulate_initial_hypothesis.return_value = mock_hyp
    
    manager = CampaignManager(mock_config)
    state = SurfaceState(bulk_composition={"Cu": 1.0}, miller_index=(1, 1, 1), termination="default")
    
    # Test planner directly
    graph = manager.strategist.planner.plan_next_steps(state, hypothesis=mock_hyp)
    task_types = [t.task_type for t in graph.nodes.values()]
    assert TaskType.RUN_MD in task_types
    assert TaskType.BUILD_SLAB in task_types

@patch('core.campaign_manager.ComputeManager')
@patch('core.campaign_manager.time.sleep', return_value=None)
def test_parallel_polling_loop(mock_sleep, mock_compute, mock_config):
    manager = CampaignManager(mock_config)
    
    # Mock strategist and its proposer
    manager.strategist.propose_actions = MagicMock(return_value=[
        (MagicMock(spec=MutationAction), MagicMock(spec=SurfaceState)),
        (MagicMock(spec=MutationAction), MagicMock(spec=SurfaceState))
    ])
    manager.strategist.score_actions = MagicMock(return_value=[0.9, 0.8])
    
    # Mock execute_best to return pending then completed
    manager.strategist.execute_best = MagicMock()
    manager.strategist.execute_best.side_effect = [
        {"status": "pending", "job_id": "j1"},
        {"status": "pending", "job_id": "j2"},
        {"status": "completed", "reward": 1.0, "observables": {}, "metadata": {"fidelity": "mock", "iteration": 1}, "state": MagicMock(), "action": MagicMock()},
        {"status": "completed", "reward": 0.8, "observables": {}, "metadata": {"fidelity": "mock", "iteration": 1}, "state": MagicMock(), "action": MagicMock()}
    ]
    
    batch = manager.strategist.propose_actions()
    results = manager._execute_batch_and_poll(batch, ".abort_test")
    
    assert len(results) == 2
    assert manager.strategist.execute_best.call_count == 4

def test_abort_signal(mock_config):
    manager = CampaignManager(mock_config)
    abort_file = ".abort"
    with open(abort_file, "w") as f:
        f.write("abort")
        
    # Should stop early
    manager.run()
    assert not os.path.exists(abort_file)

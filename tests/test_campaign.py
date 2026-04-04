import pytest
from unittest.mock import MagicMock, patch
import os
import json
from core.campaign_manager import CampaignManager
from core.state import SurfaceState
from core.lab_objects import ResearchIdea, Critique, Hypothesis, Experiment, Insight

@pytest.fixture
def mock_config():
    return {
        "name": "Test_Campaign",
        "objective": {"type": "stability"},
        "constraints": {"bulk": {"Cu": 1.0}, "facet": [1, 1, 1]},
        "optimization": {"batch_size": 1},
        "compute": {"platform": "local", "mode": "mock"},
        "budget": {"max_evaluations": 1}
    }

@patch('core.campaign_manager.ComputeManager')
@patch('memory.storage_provider.StorageRegistry')
def test_campaign_manager_init(mock_storage, mock_compute, mock_config):
    manager = CampaignManager(mock_config)
    assert manager.batch_size == 1
    assert manager.pi is not None
    assert manager.postdoc is not None
    assert manager.technician is not None

@patch('agents.postdoc_agent.genai.Client')
def test_postdoc_gatekeeping(mock_genai, mock_config):
    from agents.postdoc_agent import PostdocAgent
    from optimization.surrogate_models import GaussianProcessModel
    from memory.storage_provider import StorageRegistry
    
    storage = StorageRegistry(use_memory=False)
    # Ensure databases are initialized
    storage.experiment_db.load()
    
    postdoc = PostdocAgent(GaussianProcessModel(), storage)
    idea = ResearchIdea(goal="test", intuition="random search")
    
    # Test heuristic critique (since api_key is None in mock mode)
    critique = postdoc.review_idea(idea)
    assert critique.validity is False
    assert any("random" in issue.lower() for issue in critique.issues)

@patch('core.campaign_manager.ComputeManager')
@patch('core.campaign_manager.PIAgent')
@patch('core.campaign_manager.PostdocAgent')
@patch('core.campaign_manager.ExecutionAgent')
def test_hierarchical_loop_v5(mock_exec, mock_postdoc, mock_pi, mock_compute, mock_config):
    # Mock PI Propose
    mock_pi.return_value.propose_idea.return_value = ResearchIdea(goal="G", intuition="I")
    
    # Mock Postdoc Critique (Valid)
    mock_postdoc.return_value.review_idea.return_value = Critique(
        idea_id="1", validity=True, issues=[], revised_plan=None, confidence=0.9
    )
    
    # Mock Postdoc Hypothesis
    mock_postdoc.return_value.formulate_hypothesis.return_value = Hypothesis(
        idea_id="1", variable="v", manipulation="m", expected_effect="e", 
        metric="stability", falsification_condition="f"
    )
    
    # Mock Postdoc Design
    mock_postdoc.return_value.design_experiments.return_value = [
        Experiment(hypothesis_id="h1", parameters={}, method="MLIP", expected_output=[])
    ]
    
    # Mock Technician Execution
    real_state = SurfaceState(bulk_composition={"Cu": 1.0}, miller_index=(1, 1, 1), termination="default")
    mock_exec.return_value.run_experiments.return_value = [{
        "reward": 1.0, "state": real_state, "action": MagicMock(),
        "observables": {}, "metadata": {}, "status": "completed"
    }]
    
    # Mock Postdoc Analysis
    mock_postdoc.return_value.analyze_results.return_value = Insight(
        hypothesis_id="h1", conclusion="C", confidence=0.9
    )

    manager = CampaignManager(mock_config)
    
    # Add some initial data to avoid baseline run
    manager.storage.experiment_db.add_experiment(real_state, {"reward": 0.0}, MagicMock())
    
    manager.run()
    
    assert mock_pi.return_value.propose_idea.called
    assert mock_postdoc.return_value.review_idea.called
    assert mock_exec.return_value.run_experiments.called

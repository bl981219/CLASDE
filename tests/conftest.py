import pytest
from unittest.mock import MagicMock
from core.state import SurfaceState

@pytest.fixture
def mock_surface_state():
    return SurfaceState(
        bulk_composition={"Cu": 1.0},
        miller_index=(1, 1, 1),
        termination="default"
    )

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

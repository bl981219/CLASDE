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
def mock_storage(tmp_path):
    from memory.storage_provider import StorageRegistry
    import os
    # Point databases to temp directory
    db_dir = tmp_path / "data" / "results"
    os.makedirs(db_dir, exist_ok=True)
    
    # We can't easily override the hardcoded paths in the DB classes without refactoring them
    # But for unit tests, use_memory=False usually avoids the heavy lifting
    storage = StorageRegistry(use_memory=False)
    return storage

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

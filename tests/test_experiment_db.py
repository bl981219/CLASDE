import pytest
from core.state import SurfaceState


@pytest.fixture
def db(tmp_path):
    from memory.experiment_db import ExperimentDatabase
    return ExperimentDatabase(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def alt_state():
    """A second surface state distinct from mock_surface_state."""
    return SurfaceState(
        bulk_composition={"Fe": 1.0},
        miller_index=(1, 1, 0),
        termination="default",
    )


def test_empty_db_returns_empty_training_data(db):
    assert db.get_training_data() == []


def test_add_and_retrieve(db, mock_surface_state):
    db.add_experiment(mock_surface_state, {"reward": -2.5, "fidelity": "MLIP"})
    data = db.get_training_data()
    assert len(data) == 1
    assert data[0]["reward"] == -2.5
    assert data[0]["fidelity"] == "MLIP"


def test_get_best_reward_empty(db):
    assert db.get_best_reward() == -1e9


def test_get_best_reward_picks_maximum(db, mock_surface_state, alt_state):
    db.add_experiment(mock_surface_state, {"reward": -2.0})
    db.add_experiment(alt_state, {"reward": -1.0})
    assert db.get_best_reward() == -1.0


def test_get_by_id_round_trip(db, mock_surface_state):
    db.add_experiment(mock_surface_state, {"reward": -3.0, "fidelity": "DFT"})
    result = db.get_by_id(mock_surface_state.get_id())
    assert result is not None
    assert result["reward"] == -3.0
    assert result["fidelity"] == "DFT"


def test_duplicate_state_replaced(db, mock_surface_state):
    db.add_experiment(mock_surface_state, {"reward": -2.0})
    db.add_experiment(mock_surface_state, {"reward": -1.5})
    data = db.get_training_data()
    assert len(data) == 1
    assert data[0]["reward"] == -1.5

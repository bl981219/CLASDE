import json
import os
import pytest
from core.orchestration.checkpoint_manager import CheckpointManager


def test_checkpoint_not_exists_initially(tmp_path):
    cm = CheckpointManager(str(tmp_path / "checkpoint.json"))
    assert not cm.checkpoint_exists()


def test_save_and_load_roundtrip(tmp_path):
    cm = CheckpointManager(str(tmp_path / "checkpoint.json"))
    state = {"iteration": 5, "best_reward": -1.5, "tags": ["a", "b"]}
    cm.save_checkpoint(state)
    loaded = cm.load_checkpoint()
    assert loaded == state


def test_checkpoint_exists_after_save(tmp_path):
    cm = CheckpointManager(str(tmp_path / "checkpoint.json"))
    cm.save_checkpoint({"x": 1})
    assert cm.checkpoint_exists()


def test_load_nonexistent_returns_none(tmp_path):
    cm = CheckpointManager(str(tmp_path / "checkpoint.json"))
    assert cm.load_checkpoint() is None


def test_backup_creates_separate_file(tmp_path):
    cm = CheckpointManager(str(tmp_path / "checkpoint.json"))
    cm.save_checkpoint({"x": 1})
    backup_path = cm.backup_checkpoint(suffix="test")
    assert backup_path is not None
    assert os.path.exists(backup_path)
    assert cm.checkpoint_exists()


def test_atomic_write_no_temp_file_left(tmp_path):
    cm = CheckpointManager(str(tmp_path / "checkpoint.json"))
    cm.save_checkpoint({"x": 1})
    temp_path = str(tmp_path / "checkpoint.json.tmp")
    assert not os.path.exists(temp_path)

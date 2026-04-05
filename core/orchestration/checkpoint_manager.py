"""
Checkpoint manager for campaign state persistence.

Handles saving and loading campaign state for pause/resume functionality.
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from core.exceptions import MemoryError

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages campaign state checkpointing.
    
    Provides fault tolerance through periodic state snapshots.
    """
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = os.path.dirname(checkpoint_path)
        
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        """
        Save campaign state to disk.
        
        Args:
            state: Dictionary containing campaign state
            
        Raises:
            MemoryError: If checkpoint save fails
        """
        try:
            # Write to temporary file first (atomic write)
            temp_path = f"{self.checkpoint_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            
            # Rename to actual checkpoint (atomic on POSIX)
            os.replace(temp_path, self.checkpoint_path)
            
            logger.debug(f"Checkpoint saved to {self.checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise MemoryError(
                f"Checkpoint save failed: {e}",
                details={"checkpoint_path": self.checkpoint_path}
            )
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load campaign state from disk.
        
        Returns:
            Campaign state dictionary, or None if no checkpoint exists
            
        Raises:
            MemoryError: If checkpoint exists but is corrupted
        """
        if not self.checkpoint_exists():
            logger.debug("No checkpoint found")
            return None
        
        try:
            with open(self.checkpoint_path, "r") as f:
                state = json.load(f)
            
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
            return state
            
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted checkpoint file: {e}")
            raise MemoryError(
                f"Checkpoint file is corrupted: {e}",
                details={"checkpoint_path": self.checkpoint_path}
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise MemoryError(
                f"Checkpoint load failed: {e}",
                details={"checkpoint_path": self.checkpoint_path}
            )
    
    def checkpoint_exists(self) -> bool:
        """Check if checkpoint file exists."""
        return os.path.exists(self.checkpoint_path)
    
    def delete_checkpoint(self) -> None:
        """Remove checkpoint file."""
        if self.checkpoint_exists():
            try:
                os.remove(self.checkpoint_path)
                logger.debug(f"Checkpoint deleted: {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint: {e}")
    
    def backup_checkpoint(self, suffix: str = None) -> str:
        """
        Create backup of current checkpoint.
        
        Args:
            suffix: Optional suffix for backup filename
            
        Returns:
            Path to backup file
        """
        if not self.checkpoint_exists():
            return None
        
        import time
        if suffix is None:
            suffix = time.strftime("%Y%m%d_%H%M%S")
        
        backup_path = f"{self.checkpoint_path}.backup_{suffix}"
        
        try:
            import shutil
            shutil.copy2(self.checkpoint_path, backup_path)
            logger.info(f"Checkpoint backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup checkpoint: {e}")
            return None

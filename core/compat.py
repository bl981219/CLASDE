"""
Backward compatibility layer for CLASDE architectural refactoring.

This module provides helper functions to create campaigns using the new
architecture while maintaining the old API for existing code.
"""

import logging
import os
from typing import Dict, Any, Optional
from core.config import CampaignConfig
from core.factory import ComponentFactory, create_component_factory
from core.orchestration import CheckpointManager, TraceLogger
from core.events import EventBus, get_event_bus
from core.state_machine import CampaignStateMachine, CampaignState
from core.schemas import SystemState

logger = logging.getLogger(__name__)


def create_campaign_from_dict(
    config_dict: Dict[str, Any],
    mock: bool = False
) -> "CampaignManager":
    """
    Create a CampaignManager from a dictionary config (backward compatible).
    
    This function maintains the old API while using the new architecture internally.
    
    Args:
        config_dict: Legacy dictionary configuration
        mock: If True, use mock components for testing
        
    Returns:
        CampaignManager instance
    """
    # Convert dict to typed config
    try:
        config = CampaignConfig(**config_dict)
    except Exception as e:
        logger.warning(f"Config validation failed: {e}. Using permissive mode.")
        # Fallback to dict-based approach for backward compatibility
        from core.campaign_manager import CampaignManager
        return CampaignManager(config_dict)
    
    # Create component factory
    factory = create_component_factory(config, mock=mock)
    
    # Create orchestration components
    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    checkpoint_mgr = CheckpointManager(
        os.path.join(results_dir, "campaign_checkpoint.json")
    )
    
    trace_logger = TraceLogger(
        os.path.join(results_dir, "knowledge_trace.jsonl")
    )
    
    # Create event bus
    event_bus = get_event_bus()
    
    # Create state machine
    state_machine = CampaignStateMachine()
    
    # Create agents and components
    storage = factory.create_storage()
    surrogate = factory.create_surrogate()
    compute_manager = factory.create_compute_manager()
    
    pi = factory.create_pi_agent()
    postdoc = factory.create_postdoc_agent(surrogate, storage)
    technician = factory.create_execution_agent(compute_manager)
    governor = factory.create_governor()
    builder = factory.create_structure_builder()
    theory_builder = factory.create_theory_builder(storage)
    
    # Import and create CampaignManager with new components
    from core.campaign_manager import CampaignManager
    
    # Create with dependency injection
    manager = CampaignManager(
        config=config_dict,  # Keep as dict for now for compatibility
        storage=storage
    )
    
    # Inject new components (monkey-patch for now, refactor later)
    manager.checkpoint_mgr = checkpoint_mgr
    manager.trace_logger = trace_logger
    manager.event_bus = event_bus
    manager.state_machine = state_machine
    
    return manager


def validate_config_dict(config_dict: Dict[str, Any]) -> CampaignConfig:
    """
    Validate and upgrade legacy config dict to typed CampaignConfig.
    
    Args:
        config_dict: Legacy configuration dictionary
        
    Returns:
        Validated CampaignConfig instance
        
    Raises:
        ValidationError: If config is invalid
    """
    from core.exceptions import ConfigurationError
    
    try:
        return CampaignConfig(**config_dict)
    except Exception as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}",
            details={"config": config_dict}
        )


# Convenience functions for creating specific components

def create_llm_provider(mock: bool = False):
    """Create LLM provider."""
    if mock:
        from unittest.mock import Mock
        llm = Mock()
        llm.generate.return_value = {"idea": "Test idea", "confidence": 0.9}
        return llm
    
    from agents.collaborator_agent import LLMCollaborator
    return LLMCollaborator()


def create_storage_registry(auto_load: bool = True):
    """Create storage registry."""
    from memory.storage_provider import StorageRegistry
    storage = StorageRegistry()
    if auto_load and not storage.is_loaded:
        storage.load_all()
    return storage


def create_surrogate_model(model_type: str = "gp"):
    """Create surrogate model."""
    if model_type == "gp":
        from optimization.surrogate_models import GaussianProcessModel
        return GaussianProcessModel()
    else:
        raise ValueError(f"Unknown surrogate model type: {model_type}")


def create_execution_backend(backend_type: str = "local", config: Dict = None):
    """Create execution backend."""
    config = config or {}
    
    if backend_type == "slurm":
        from execution.backends import SlurmBackend
        return SlurmBackend(config)
    elif backend_type == "local":
        from execution.backends import LocalBackend
        return LocalBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

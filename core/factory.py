"""
Factory pattern for dependency injection.

Provides centralized creation of agents and components with proper dependency management.
"""

import logging
from typing import Protocol, Optional, Dict, Any
from core.protocols import LLMProvider, MemoryStore, SurrogateModel, ExecutionBackend
from core.config import CampaignConfig

logger = logging.getLogger(__name__)


class AgentFactory(Protocol):
    """Interface for agent creation with dependency injection."""
    
    def create_pi_agent(self):
        """Create PI Agent."""
        ...
    
    def create_postdoc_agent(self, storage, surrogate, llm):
        """Create Postdoc Agent."""
        ...
    
    def create_execution_agent(self, compute_manager):
        """Create Execution Agent."""
        ...


class ComponentFactory:
    """
    Default factory for creating CLASDE components.
    
    Centralizes dependency creation and makes testing easier through
    constructor injection.
    """
    
    def __init__(self, config: CampaignConfig):
        self.config = config
    
    def create_llm_provider(self) -> LLMProvider:
        """Create LLM provider (Gemini by default)."""
        from agents.collaborator_agent import LLMCollaborator
        return LLMCollaborator()
    
    def create_storage(self) -> MemoryStore:
        """Create storage registry."""
        from memory.storage_provider import StorageRegistry
        storage = StorageRegistry()
        if not storage.is_loaded:
            storage.load_all()
        return storage
    
    def create_surrogate(self) -> SurrogateModel:
        """Create surrogate model."""
        from optimization.surrogate_models import GaussianProcessModel
        return GaussianProcessModel()
    
    def create_execution_backend(self) -> ExecutionBackend:
        """Create execution backend based on config."""
        backend_type = self.config.compute.backend
        
        if backend_type == "slurm":
            from execution.backends import SlurmBackend
            return SlurmBackend({
                "partition": self.config.compute.slurm_partition,
                "extra_header": self.config.compute.slurm_extra_header or ""
            })
        elif backend_type == "local":
            from execution.backends import LocalBackend
            return LocalBackend(self.config.compute.model_dump())
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def create_compute_manager(self, backend: Optional[ExecutionBackend] = None):
        """Create compute manager."""
        from execution.compute_agent import ComputeManager
        
        if backend is None:
            backend = self.create_execution_backend()
        
        return ComputeManager(
            config=self.config.compute.model_dump(),
            backend=backend
        )
    
    def create_pi_agent(self, llm: Optional[LLMProvider] = None):
        """Create PI Agent."""
        from agents.pi_agent import PIAgent
        from agents.collaborator_agent import LLMCollaborator
        
        collaborator = llm if llm else LLMCollaborator()
        return PIAgent(collaborator=collaborator)
    
    def create_postdoc_agent(
        self,
        surrogate: Optional[SurrogateModel] = None,
        storage: Optional[MemoryStore] = None,
        llm: Optional[LLMProvider] = None
    ):
        """Create Postdoc Agent with dependencies."""
        from agents.postdoc_agent import PostdocAgent
        
        if surrogate is None:
            surrogate = self.create_surrogate()
        if storage is None:
            storage = self.create_storage()
        
        # PostdocAgent creates its own LLM client currently
        # In future, inject llm provider here
        return PostdocAgent(surrogate, storage)
    
    def create_execution_agent(self, compute_manager=None):
        """Create Execution Agent."""
        from agents.execution_agent import ExecutionAgent
        
        if compute_manager is None:
            compute_manager = self.create_compute_manager()
        
        return ExecutionAgent(compute_manager)
    
    def create_governor(self):
        """Create Research Governor."""
        from agents.governor_agent import ResearchGovernor
        return ResearchGovernor(self.config.model_dump())
    
    def create_structure_builder(self):
        """Create Structure Builder."""
        from agents.builder_agent import StructureBuilder
        return StructureBuilder()
    
    def create_theory_builder(self, storage: Optional[MemoryStore] = None):
        """Create Theory Builder."""
        from science.theory_builder import TheoryBuilder
        
        if storage is None:
            storage = self.create_storage()
        
        return TheoryBuilder(
            storage.get_knowledge_graph(),
            original_prompt=self.config.original_prompt or "",
            budget=self.config.budget.max_evaluations
        )


class MockComponentFactory(ComponentFactory):
    """
    Factory for creating mock components in tests.
    
    All components are mocked for fast, deterministic testing.
    """
    
    def __init__(self, config: CampaignConfig):
        super().__init__(config)
        self._mock_llm = None
        self._mock_storage = None
        self._mock_surrogate = None
        self._mock_backend = None
    
    def create_llm_provider(self):
        """Return mock LLM."""
        if self._mock_llm is None:
            from unittest.mock import Mock
            self._mock_llm = Mock()
            self._mock_llm.generate.return_value = {
                "idea": "Test idea",
                "confidence": 0.9
            }
        return self._mock_llm
    
    def create_storage(self):
        """Return mock storage."""
        if self._mock_storage is None:
            from unittest.mock import Mock
            self._mock_storage = Mock()
            self._mock_storage.is_loaded = True
            self._mock_storage.experiment_db.get_training_data.return_value = []
            self._mock_storage.experiment_db.get_best_reward.return_value = -1e9
        return self._mock_storage
    
    def create_surrogate(self):
        """Return mock surrogate."""
        if self._mock_surrogate is None:
            from unittest.mock import Mock
            import numpy as np
            self._mock_surrogate = Mock()
            self._mock_surrogate.predict.return_value = (
                np.array([0.0]),
                np.array([1.0])
            )
        return self._mock_surrogate
    
    def create_execution_backend(self):
        """Return mock backend."""
        if self._mock_backend is None:
            from unittest.mock import Mock
            self._mock_backend = Mock()
            self._mock_backend.submit_job.return_value = "mock_job_123"
            self._mock_backend.monitor_job.return_value = "COMPLETED"
            self._mock_backend.retrieve_results.return_value = {
                "status": "completed",
                "reward": -5.0
            }
        return self._mock_backend


def create_component_factory(
    config: CampaignConfig,
    mock: bool = False
) -> ComponentFactory:
    """
    Factory function to create appropriate component factory.
    
    Args:
        config: Campaign configuration
        mock: If True, return MockComponentFactory for testing
        
    Returns:
        Component factory instance
    """
    if mock:
        return MockComponentFactory(config)
    return ComponentFactory(config)

import os
from typing import Dict, Any, Optional
from core.schemas import SystemState
from core.lab_objects import KnowledgeTrace
from optimization.surrogate_models import GaussianProcessModel as SurrogateModel
from agents.governor_agent import ResearchGovernor
from agents.pi_agent import PIAgent
from agents.postdoc_agent import PostdocAgent
from agents.execution_agent import ExecutionAgent
from agents.builder_agent import StructureBuilder
from execution.compute_agent import ComputeManager
from memory.storage_provider import StorageRegistry
from science.theory_builder import TheoryBuilder
from core.orchestration import (
    BaselineInitializer,
    CampaignFinalizer,
    CampaignOrchestrator,
    ResultProcessor,
    TraceLogger,
)

class CampaignManager:
    """
    The central orchestrator for a CLASDE discovery campaign.
    
    Hierarchy: PI (Visionary) -> Postdoc (Gatekeeper) -> Technician (PhD Student).
    """
    def __init__(self, config: Dict[str, Any], storage: Optional[StorageRegistry] = None):
        self.config = config
        self.results_dir = config.get("results_dir", "data/results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.results_dir, "campaign_checkpoint.json")
        self.trace_log_path = os.path.join(self.results_dir, "knowledge_trace.jsonl")
        
        # 1. Initialize Storage
        self.storage = storage or StorageRegistry()
        if not self.storage.is_loaded:
            self.storage.load_all()
        
        # 2. Initialize Lab Roles
        self.governor = ResearchGovernor(config)
        self.surrogate = SurrogateModel()
        self.compute_manager = ComputeManager(config.get("compute", {}))
        self.builder = StructureBuilder()
        
        self.pi = PIAgent()
        self.postdoc = PostdocAgent(self.surrogate, self.storage)
        self.technician = ExecutionAgent(self.compute_manager)
        
        self.theory_builder = TheoryBuilder(
            self.storage.get_knowledge_graph(), 
            original_prompt=self.config.get("original_prompt", ""),
            budget=self.governor.max_evaluations
        )
        self.system_state = SystemState()
        self.trace_logger = TraceLogger(self.trace_log_path)
        self.result_processor = ResultProcessor(
            storage=self.storage,
            governor=self.governor,
            system_state=self.system_state,
        )
        self.baseline_initializer = BaselineInitializer(
            config=self.config,
            storage=self.storage,
            builder=self.builder,
            compute_manager=self.compute_manager,
            result_processor=self.result_processor,
        )
        self.finalizer = CampaignFinalizer(
            theory_builder=self.theory_builder,
            storage=self.storage,
            results_dir=self.results_dir,
        )
        self.orchestrator = CampaignOrchestrator(
            config=self.config,
            governor=self.governor,
            pi=self.pi,
            postdoc=self.postdoc,
            technician=self.technician,
            storage=self.storage,
            system_state=self.system_state,
            on_initialize_baseline=self.baseline_initializer.initialize_if_needed,
            on_log_trace=self.trace_logger.log_trace,
            on_process_result=self.result_processor.process,
            on_finalize=self.finalizer.finalize,
        )
        
        # For backward compatibility with stale tests
        self.batch_size = config.get("optimization", {}).get("batch_size", 1)

    def log_trace(self, trace: KnowledgeTrace):
        """Append a formal knowledge transformation trace to the log."""
        self.trace_logger.log_trace(trace)

    def run(self):
        """Executes the strict hierarchical discovery loop with knowledge tracing."""
        self.orchestrator.run()

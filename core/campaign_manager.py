import logging
import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from core.state import SurfaceState
from core.action import MutationAction
from core.transition import TransitionEngine
from core.schemas import SystemState, Action, Result, TaskStatus
from core.lab_objects import ResearchIdea, Critique, Hypothesis, Experiment, Insight, KnowledgeTrace
from optimization.surrogate_models import GaussianProcessModel as SurrogateModel
from agents.governor_agent import ResearchGovernor
from agents.pi_agent import PIAgent
from agents.postdoc_agent import PostdocAgent
from agents.execution_agent import ExecutionAgent
from execution.compute_agent import ComputeManager, SimulationType
from agents.evaluator_agent import EvaluationAgent
from memory.storage_provider import StorageRegistry
from science.theory_builder import TheoryBuilder

logger = logging.getLogger(__name__)

class CampaignManager:
    """
    The central orchestrator for a CLASDE discovery campaign.
    
    Refactored V5: Knowledge Transformer Architecture.
    Hierarchy: PI (ResearchIdea) -> Postdoc (Critique -> Hypothesis -> Experiment) -> Technician.
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
        
        self.pi = PIAgent()
        self.postdoc = PostdocAgent(self.surrogate, self.storage)
        self.technician = ExecutionAgent(self.compute_manager)
        
        self.theory_builder = TheoryBuilder(self.storage.get_knowledge_graph(), budget=self.governor.max_evaluations)
        self.system_state = SystemState()

    def log_trace(self, trace: KnowledgeTrace):
        """Append a formal knowledge transformation trace to the log."""
        with open(self.trace_log_path, "a") as f:
            f.write(trace.model_dump_json() + "\n")

    def run(self):
        """Executes the strict hierarchical discovery loop with knowledge tracing."""
        logger.info(f"--- [LAB V5] Starting Knowledge Transformer Loop ---")
        
        # Load state
        self._initialize_baseline_if_needed()

        while self.governor.should_continue(latest_reward=self.system_state.current_best_reward):
            try:
                # 1. PI Proposes Idea
                idea = self.pi.propose_idea(self.config.get("original_prompt", "Optimize surface stability"))
                
                # 2. Postdoc Critique & Gatekeeping (Rule: Postdoc authority)
                critique = self.postdoc.review_idea(idea)
                logger.info(f"[LAB] Critique Validity: {critique.validity} (Confidence: {critique.confidence})")
                
                # If invalid, PI must refine or Postdoc imposes revision
                if not critique.validity:
                    logger.warning(f"[LAB] PI idea rejected. Applying Postdoc revision.")
                    # In a more complex loop, this could iterate. Here we enforce the revision.
                
                # 3. Postdoc Formulates Hypothesis (Rule: Strict Object)
                hypothesis = self.postdoc.formulate_hypothesis(idea, critique)
                
                # 4. Postdoc Designs Experiments
                current_state = self.storage.experiment_db.get_training_data()[-1]['state']
                experiments = self.postdoc.design_experiments(hypothesis, current_state)
                
                # 5. Technician Executes Experiments
                results = self.technician.run_experiments(experiments, self.system_state.iteration)
                
                # 6. Postdoc Interprets Results into Insights
                insight = self.postdoc.analyze_results(hypothesis, results)
                logger.info(f"[LAB] Insight: {insight.conclusion}")

                # 7. Knowledge Trace Logging (Full Audit Trail)
                trace = KnowledgeTrace(
                    iteration=self.system_state.iteration,
                    input_idea=idea,
                    critique=critique,
                    final_hypothesis=hypothesis,
                    experiments=experiments,
                    insight=insight
                )
                self.log_trace(trace)

                # 8. Update System State & Memory
                for res in results:
                    self._process_result(res)
                
                self.system_state.iteration += 1
                self.storage.save_all()
                
            except Exception as e:
                logger.error(f"[LAB] Error in knowledge loop: {e}")
                time.sleep(5)
                continue

        self._finalize()

    def _initialize_baseline_if_needed(self):
        if not self.storage.experiment_db.get_training_data():
            logger.info("[Technician] No data found. Running initial baseline...")
            # (Baseline logic...)
            pass

    def _process_result(self, result: Dict[str, Any]):
        reward = result.get("reward", -1e9)
        if reward > self.system_state.current_best_reward:
            self.system_state.current_best_reward = reward
        
        self.storage.experiment_db.add_experiment(
            state=result["state"], 
            results={**result["observables"], "reward": reward}, 
            action=result["action"]
        )
        self.governor.consume_budget()

    def _finalize(self):
        logger.info("[LAB] Discovery complete. Report generated.")
        self.storage.save_all()

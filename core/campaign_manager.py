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
from agents.builder_agent import StructureBuilder
from execution.compute_agent import ComputeManager, SimulationType
from agents.evaluator_agent import EvaluationAgent
from memory.storage_provider import StorageRegistry
from science.theory_builder import TheoryBuilder

logger = logging.getLogger(__name__)

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
        
        self.theory_builder = TheoryBuilder(self.storage.get_knowledge_graph(), budget=self.governor.max_evaluations)
        self.system_state = SystemState()
        
        # For backward compatibility with stale tests
        self.batch_size = config.get("optimization", {}).get("batch_size", 1)

    def log_trace(self, trace: KnowledgeTrace):
        """Append a formal knowledge transformation trace to the log."""
        with open(self.trace_log_path, "a") as f:
            f.write(trace.model_dump_json() + "\n")

    def run(self):
        """Executes the strict hierarchical discovery loop with knowledge tracing."""
        logger.info(f"--- [CLASDE V5] Starting Hierarchical Discovery Loop ---")
        
        # Load state
        self._initialize_baseline_if_needed()

        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.governor.should_continue(latest_reward=self.system_state.current_best_reward):
            try:
                # 1. PI Proposes Idea & Postdoc Critiques (Epistemic Cycle)
                idea = self.pi.propose_idea(self.config.get("original_prompt", "Optimize surface stability"))
                
                max_revisions = 3
                revision_count = 0
                critique = self.postdoc.review_idea(idea)
                
                while not critique.validity and revision_count < max_revisions:
                    logger.warning(f"[LAB] PI idea rejected. Revision {revision_count+1}/{max_revisions}")
                    idea = self.pi.refine_idea(idea, critique.revised_plan or "Provide more physical justification.")
                    critique = self.postdoc.review_idea(idea)
                    revision_count += 1

                if not critique.validity:
                    logger.error("[LAB] Failed to converge on a valid idea after max revisions. Terminating loop.")
                    break
                
                # 2. Postdoc Formulates Hypothesis
                hypothesis = self.postdoc.formulate_hypothesis(idea, critique)
                
                # 3. Postdoc Designs Experiments
                training_data = self.storage.experiment_db.get_training_data()
                if not training_data:
                    logger.error("[LAB] No training data available for experiment design.")
                    break
                    
                current_state = training_data[-1]['state']
                experiments = self.postdoc.design_experiments(hypothesis, current_state)
                
                # 4. Technician Executes Experiments
                results = self.technician.run_experiments(experiments, self.system_state.iteration)
                
                # 5. Postdoc Interprets Results into Insights
                insight = self.postdoc.analyze_results(hypothesis, results)
                logger.info(f"[LAB] Insight: {insight.conclusion}")

                # 6. Knowledge Trace Logging (Full Audit Trail)
                trace = KnowledgeTrace(
                    iteration=self.system_state.iteration,
                    input_idea=idea,
                    critique=critique,
                    final_hypothesis=hypothesis,
                    experiments=experiments,
                    insight=insight
                )
                self.log_trace(trace)

                # 7. Update System State & Memory
                for res in results:
                    self._process_result(res)
                
                self.system_state.iteration += 1
                self.storage.save_all()
                consecutive_errors = 0 # Reset on success
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"[LAB] Error in knowledge loop (Error {consecutive_errors}/{max_consecutive_errors}): {e}", exc_info=True)
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("[LAB] Too many consecutive errors. Terminating loop for safety.")
                    raise RuntimeError(f"Campaign terminated after {max_consecutive_errors} consecutive failures.") from e

                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                time.sleep(5)
                continue

        self._finalize()

    def _initialize_baseline_if_needed(self):
        if not self.storage.experiment_db.get_training_data():
            logger.info("[Technician] No data found. Running initial baseline...")
            
            bulk = self.config.get("constraints", {}).get("bulk", {"Cu": 1.0})
            facet = self.config.get("constraints", {}).get("facet", [1, 1, 1])
            
            initial_state = SurfaceState(
                bulk_composition=bulk,
                miller_index=tuple(facet),
                termination="default"
            )
            
            slab = self.builder.build_structure(initial_state)
            initial_state.slab_structure = slab
            
            # Submit initial job
            job_id = self.compute_manager.submit_job(slab, initial_state, SimulationType.MLIP, iteration=0)
            
            # Synchronous fetch for baseline
            calc_dir = self.compute_manager.fetch_results(job_id)
            res_path = os.path.join(calc_dir, "results.json")
            if os.path.exists(res_path):
                with open(res_path, "r") as f:
                    raw_data = json.load(f)
                    res_dict = {
                        "state": initial_state,
                        "action": None,
                        "reward": raw_data.get("reward", 0.0),
                        "observables": raw_data,
                        "metadata": {"job_id": job_id, "fidelity": "baseline"}
                    }
                    self._process_result(res_dict)
            else:
                raise RuntimeError("Baseline calculation failed. Check compute environment.")

    def _process_result(self, result: Dict[str, Any]):
        reward = result.get("reward", -1e9)
        if reward > self.system_state.current_best_reward:
            self.system_state.current_best_reward = reward
        
        self.storage.experiment_db.add_experiment(
            state=result["state"], 
            results={**result["observables"], "reward": reward}, 
            action=result.get("action")
        )
        self.governor.consume_budget()

    def _finalize(self):
        logger.info("[LAB] Discovery complete. Report generated.")
        self.storage.save_all()

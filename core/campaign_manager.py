import logging
import os
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from core.state import SurfaceState
from core.action import MutationAction
from core.transition import TransitionEngine
from core.schemas import SystemState, Action, Result, TaskStatus
from core.lab_objects import ResearchIdea, Critique, Hypothesis, Experiment, Insight
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
    
    Refactored V4: High-Performing Lab Analogy.
    Information Flow: PI (Idea) -> Postdoc (Critique/Hypothesis/Design) -> Technician (Execution).
    """
    def __init__(self, config: Dict[str, Any], storage: Optional[StorageRegistry] = None):
        self.config = config
        self.results_dir = config.get("results_dir", "data/results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.results_dir, "campaign_checkpoint.json")
        self.log_file = os.path.join(self.results_dir, "research_log.md")
        
        # 1. Initialize Storage & Memory
        self.storage = storage or StorageRegistry()
        if not self.storage.is_loaded:
            self.storage.load_all()
        
        self.experiment_db = self.storage.experiment_db
        self.hypothesis_db = self.storage.hypothesis_db
        self.knowledge_graph = self.storage.get_knowledge_graph()
        
        # 2. Initialize Lab Staff
        self.governor = ResearchGovernor(config)
        self.surrogate = SurrogateModel()
        self.compute_manager = ComputeManager(config.get("compute", {}))
        
        self.pi = PIAgent()
        self.postdoc = PostdocAgent(self.surrogate, self.experiment_db, self.knowledge_graph)
        self.technician = ExecutionAgent(self.compute_manager)
        
        self.evaluator = EvaluationAgent(self.governor.get_reward_function(), self.knowledge_graph)
        self.theory_builder = TheoryBuilder(self.knowledge_graph, budget=self.governor.max_evaluations)
        
        self.system_state = SystemState()

    def save_checkpoint(self):
        """Persists the current system state to disk."""
        with open(self.checkpoint_path, "w") as f:
            f.write(self.system_state.model_dump_json())

    def load_checkpoint(self) -> bool:
        """Restores system state from disk if available."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    self.system_state = SystemState.model_validate_json(f.read())
                return True
            except: pass
        return False

    def run(self):
        """Executes the hierarchical lab discovery loop."""
        logger.info(f"--- [LAB] Campaign {self.config.get('name')} Started ---")
        
        if not self.load_checkpoint():
            self._initialize_log()
            # 1. PI Proposes Idea
            idea = self.pi.propose_idea(self.config.get("original_prompt", "Optimize surface stability"))
            
            # 2. Postdoc Critiques PI (Rule 2: Debate step)
            critique = self.postdoc.critique_idea(idea)
            if not critique.validity:
                idea = self.pi.refine_idea(idea, critique.suggested_revision)
            
            # 3. Baseline Calculation
            if not self.experiment_db.get_training_data():
                self._run_baseline()
            
            self.save_checkpoint()

        # 4. Discovery Loop
        while self.governor.should_continue(latest_reward=self.system_state.current_best_reward, current_uncertainty=1.0):
            try:
                # A. Observations & Belief Update
                obs = self.postdoc.observe_state()
                self.postdoc.update_belief(obs)
                
                # B. Postdoc Formalizes Hypothesis (Rule 3)
                current_idea = ResearchIdea(goal=self.config.get("name"), intuition=self.config.get("original_prompt"))
                hypothesis = self.postdoc.formulate_hypothesis(current_idea)
                
                # C. Postdoc Designs Experiments (Rule 4)
                current_state = self.experiment_db.get_training_data()[-1]['state']
                experiments = self.postdoc.design_experiments(hypothesis, current_state)
                
                # D. Technician Executes Experiments
                results = self.technician.run_experiments(experiments, self.system_state.iteration)
                
                # E. Postdoc Interprets Results into Insights
                insight = self.postdoc.interpret_results(hypothesis, experiments, results)
                logger.info(f"[LAB] Insight: {insight.conclusion}")

                # F. Update Memory & State
                for res in results:
                    self._process_result(res)
                
                self.system_state.iteration += 1
                self.save_checkpoint()
                self.storage.save_all()
                
            except Exception as e:
                logger.error(f"[LAB] Error in discovery cycle: {e}")
                time.sleep(5)
                continue

        self._finalize()

    def _process_result(self, result: Dict[str, Any]):
        reward = result.get("reward", -1e9)
        if reward > self.system_state.current_best_reward:
            self.system_state.current_best_reward = reward
        
        # Log to DBs
        self.experiment_db.add_experiment(
            state=result["state"], 
            results={**result["observables"], "reward": reward}, 
            action=result["action"]
        )
        self.knowledge_graph.record_experiment(
            result["state"], result["action"], 
            result["observables"], result["metadata"]
        )
        self.governor.consume_budget()

    def _initialize_log(self):
        with open(self.log_file, "w") as f:
            f.write(f"# Lab Campaign Log: {self.config.get('name')}\n")
            f.write("| Iter | Idea/Hypothesis | Status | Reward | Best |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    def _run_baseline(self):
        logger.info("[Technician] Establishing pristine baseline...")
        # (Simplified baseline logic)
        current_state = SurfaceState(
            bulk_composition=self.config["constraints"]["bulk"],
            miller_index=tuple(self.config["constraints"]["facet"]),
            termination="default"
        )
        from agents.builder_agent import StructureBuilder
        builder = StructureBuilder()
        slab = builder.build_structure(current_state)
        job_id = self.compute_manager.submit_job(slab, current_state, SimulationType.MLIP, iteration=0)
        
    def _finalize(self):
        logger.info("[LAB] Campaign Finalized. Generating Discovery Report.")
        self.theory_builder.generate_report()
        self.storage.save_all()

import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from core.state import SurfaceState
from core.action import MutationAction
from core.transition import TransitionEngine
from optimization.surrogate_models import GaussianProcessModel as SurrogateModel
from agents.governor_agent import ResearchGovernor
from agents.strategist_agent import OptimizationStrategist
from agents.builder_agent import StructureBuilder
from execution.compute_agent import ComputeManager
from agents.evaluator_agent import EvaluationAgent
from memory.experiment_db import ExperimentDatabase
from memory.hypothesis_db import HypothesisDatabase
from memory.literature_db import LiteratureDatabase
from memory.knowledge_graph import KnowledgeGraphMemory
from memory.storage_provider import StorageRegistry
from agents.collaborator_agent import LLMCollaborator
from agents.hypothesis_agent import HypothesisAgent
from science.theory_builder import TheoryBuilder

logger = logging.getLogger(__name__)

class CampaignManager:
    """
    The central orchestrator for a CLASDE discovery campaign.
    
    This manager holds the state of the research (databases, agents, and configuration)
    and executes the agentic discovery loop:
    Observe -> Update Belief -> Propose -> Plan -> Execute -> Evaluate -> Verify/Evolve Hypothesis.
    """
    def __init__(self, config: Dict[str, Any], storage: Optional[StorageRegistry] = None):
        self.config = config
        self.results_dir = "data/results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.log_file = os.path.join(self.results_dir, "research_log.md")
        
        # 1. Initialize Storage & Memory
        self.storage = storage or StorageRegistry()
        if not self.storage.is_loaded:
            self.storage.load_all()
        
        self.experiment_db = self.storage.experiment_db
        self.hypothesis_db = self.storage.hypothesis_db
        self.literature_db = self.storage.literature_db
        self.knowledge_graph = self.storage.get_knowledge_graph()
        
        # 2. Initialize Agents
        self.governor = ResearchGovernor(config)
        self.surrogate = SurrogateModel()
        self.builder = StructureBuilder()
        self.compute = ComputeManager(config.get("compute", {}))
        self.evaluator = EvaluationAgent(self.governor.get_reward_function(), self.knowledge_graph)
        
        self.pi_agent = HypothesisAgent(self.knowledge_graph, self.hypothesis_db)
        self.theory_builder = TheoryBuilder(
            self.knowledge_graph, 
            original_prompt=config.get("original_prompt", ""),
            budget=self.governor.max_evaluations
        )
        
        self.strategist = OptimizationStrategist(
            surrogate=self.surrogate, 
            config=config, 
            experiment_db=self.experiment_db,
            compute_manager=self.compute,
            builder=self.builder,
            evaluator=self.evaluator,
            knowledge_graph=self.knowledge_graph,
            hypothesis_db=self.hypothesis_db
        )
        
        self.current_hypothesis = None
        self.batch_size = config.get("optimization", {}).get("batch_size", 1)

    @staticmethod
    def from_prompt(prompt: str, overrides: Optional[Dict[str, Any]] = None) -> 'CampaignManager':
        """
        Bootstrap a campaign directly from a natural language prompt.
        """
        collaborator = LLMCollaborator()
        config = collaborator.translate_goal_to_campaign(prompt)
        config["original_prompt"] = prompt
        if overrides:
            config.update(overrides)
        return CampaignManager(config)

    def run(self):
        """Executes the discovery loop with parallel batch discovery and stateful control."""
        logger.info(f"--- Campaign {self.config.get('name', 'Unnamed')} Started ---")
        self._initialize_log()
        
        # 1. Initial Hypothesis
        self.current_hypothesis = self._formulate_initial_hypothesis()
        
        # 2. Baseline Calculation
        if not self.experiment_db.get_training_data():
            self._run_baseline()
            
        # 3. Discovery Loop
        reward = -1e9
        sigma = 1.0
        abort_file = ".abort"
        
        while self.governor.should_continue(latest_reward=reward, current_uncertainty=sigma):
            if os.path.exists(abort_file):
                logger.warning(f"Abort signal detected. Stopping campaign gracefully.")
                os.remove(abort_file)
                break
            
            try:
                # A. Strategic Observation & Update
                obs = self.strategist.observe_state()
                self.strategist.update_belief(obs)
                
                # B. Propose Batch of Candidates
                candidates = self.strategist.propose_actions()
                scores = self.strategist.score_actions(candidates)
                
                import numpy as np
                best_indices = np.argsort(scores)[-self.batch_size:][::-1]
                batch = [candidates[i] for i in best_indices]
                
                # D. Parallel Execution & Polling
                results = self._execute_batch_and_poll(batch, abort_file)
                if not results: return # Interrupted

                # E. Update Memory & Evolution
                for result in results:
                    if result.get("status") == "failed":
                        logger.error(f"Task failed: {result.get('error')}. Recording failure.")
                        reward = -1e10 # Severe penalty
                        result["reward"] = reward
                        if "observables" not in result: result["observables"] = {}
                    else:
                        reward = result.get("reward", -1e9)

                    sigma = result.get("metadata", {}).get("sigma", 1.0)

                    
                    self.strategist.update_memory(result)
                    self._log_iteration(result)
                    self.knowledge_graph.record_experiment(result["state"], result["action"], result["observables"], result["metadata"])
                    self.governor.consume_budget()

                self.storage.save_all()
                self._evolve_hypothesis()
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(10)
                continue

        self._finalize()

    def _execute_batch_and_poll(self, batch: List[Tuple[MutationAction, SurfaceState]], abort_file: str) -> List[Dict[str, Any]]:
        """Submits a batch of actions and polls until all are complete."""
        in_flight = []
        for action_tuple in batch:
            res = self.strategist.execute_best(action_tuple, hypothesis=self.current_hypothesis)
            in_flight.append((action_tuple, res))
            
        completed_results = []
        while len(in_flight) > 0:
            if os.path.exists(abort_file): return []
            
            remaining = []
            for action_tuple, res in in_flight:
                if res.get("status") == "pending":
                    # Re-poll
                    updated_res = self.strategist.execute_best(action_tuple, hypothesis=self.current_hypothesis)
                    if updated_res.get("status") == "pending":
                        remaining.append((action_tuple, updated_res))
                    else:
                        completed_results.append(updated_res)
                else:
                    completed_results.append(res)
            
            if remaining:
                logger.info(f"Polling {len(remaining)} in-flight jobs... Sleeping 60s")
                time.sleep(60)
                in_flight = remaining
            else:
                in_flight = []
                
        return completed_results

    def _initialize_log(self):
        with open(self.log_file, "w") as f:
            f.write(f"\n# Research Campaign: {self.config.get('name', 'Unnamed')}\n")
            f.write("## Exploration Phase\n")
            f.write("| Iteration | Action | Fidelity | Reward | Best Reward |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    def _formulate_initial_hypothesis(self):
        claims = []
        if self.config.get("literature_check", False):
            bulk = self.config.get("constraints", {}).get("bulk", {})
            claims = self.literature_db.find_claims(list(bulk.keys()))
            
        hypothesis = self.pi_agent.formulate_initial_hypothesis(claims, self.config.get("original_prompt", ""))
        logger.info(f"--- Initial PI Hypothesis: {hypothesis.theory_statement} ---")
        self.theory_builder.add_hypothesis_record(hypothesis, "Initial Formulation")
        return hypothesis

    def _run_baseline(self):
        logger.info("Establishing pristine baseline...")
        # (Simplified baseline logic, can be made more generic)
        current_state = SurfaceState(
            bulk_composition=self.config["constraints"]["bulk"],
            miller_index=tuple(self.config["constraints"]["facet"]),
            termination="default"
        )
        slab = self.builder.build_structure(current_state)
        current_state.slab_structure = slab
        
        job_id = self.compute.submit_job(slab, current_state, sim_type="mlip", iteration=0)
        results_dir = self.compute.fetch_results(job_id)
        observables, reward = self.evaluator.evaluate_calculation(results_dir, {"state": current_state})
        
        init_data = {"reward": reward, "fidelity": "mlip", **observables}
        self.experiment_db.add_experiment(current_state, init_data)
        self.knowledge_graph.record_experiment(current_state, None, init_data)

    def _log_iteration(self, result: Dict[str, Any]):
        metadata = result["metadata"]
        with open(self.log_file, "a") as f:
            f.write(f"| {metadata['iteration']} | {result['action'].action_type.value} | {metadata['fidelity'].upper()} | {result['reward']:.4f} | {self.experiment_db.get_best_reward():.4f} |\n")

    def _evolve_hypothesis(self):
        verification_msg = self.pi_agent.verify_current_hypothesis(self.current_hypothesis, self.experiment_db.get_training_data())
        logger.info(f"[PI Verification] {verification_msg}")
        self.theory_builder.add_hypothesis_record(self.current_hypothesis, verification_msg)
        
        self.current_hypothesis = self.pi_agent.evolve_hypothesis(self.current_hypothesis, self.experiment_db.get_training_data())
        logger.info(f"--- New Hypothesis: {self.current_hypothesis.theory_statement} ---")

    def _finalize(self):
        self.theory_builder.identify_electronic_descriptors()
        report = self.theory_builder.generate_report()
        report_path = os.path.join(self.results_dir, "discovery_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        self.storage.save_all()
        logger.info(f"Campaign Finalized. Report saved to {report_path}")

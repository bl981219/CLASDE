import logging
import os
import time
from typing import List, Tuple, Dict, Any, Optional

import yaml
from core.state import SurfaceState, AdsorbateInstance
from core.action import MutationAction, ActionType
from core.transition import TransitionEngine
from optimization.surrogate_models import GaussianProcessModel as SurrogateModel
from agents.governor_agent import ResearchGovernor
from agents.strategist_agent import OptimizationStrategist
from agents.builder_agent import StructureBuilder
from execution.compute_agent import ComputeManager, SimulationType
from agents.evaluator_agent import EvaluationAgent
from memory.experiment_db import ExperimentDatabase
from memory.hypothesis_db import HypothesisDatabase
from memory.literature_db import LiteratureDatabase
from memory.knowledge_graph import KnowledgeGraphMemory
from science.experiment_graph import KnowledgeGraph
from agents.hypothesis_agent import HypothesisAgent
from science.theory_builder import TheoryBuilder

# Configure logger
logger = logging.getLogger(__name__)

class ReproducibilityLayer:
    def capture_environment(self) -> Dict[str, Any]:
        import sys
        import platform
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": time.ctime(),
            "random_seed": 42
        }

def run_adsorption_campaign(config: Dict[str, Any]) -> None:
    """
    Orchestrate the high-level CLASDE Bayesian Optimization (BO) loop.

    Initializes all agents, databases, and HPC managers based on the provided configuration.
    Executes a continuous loop of Observe -> Update Belief -> Propose -> Execute -> Evaluate 
    until the defined computational budget is exhausted. Also triggers autonomous scientific 
    reasoning at the end of the campaign.

    Args:
        config (Dict[str, Any]): A dictionary containing the campaign parameters 
                                 (objective, budget, constraints, compute environment).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    repro = ReproducibilityLayer()
    env_metadata = repro.capture_environment()
    
    if "constraints" in config and "facet" in config["constraints"]:
        config["constraints"]["facet"] = tuple(config["constraints"]["facet"])

    # 1. Component Initialization
    governor = ResearchGovernor(config)
    experiment_db = ExperimentDatabase()
    hypothesis_db = HypothesisDatabase()
    literature_db = LiteratureDatabase()
    kg_memory = KnowledgeGraphMemory()
    
    experiment_db.load()
    hypothesis_db.load()
    literature_db.load()
    knowledge_graph = kg_memory.load()
    
    pi_agent = HypothesisAgent(knowledge_graph, hypothesis_db)
    theory_builder = TheoryBuilder(knowledge_graph, original_prompt=config.get("original_prompt", ""))
    
    results_dir: str = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    log_file: str = os.path.join(results_dir, "research_log.md")
    
    with open(log_file, "w") as f:
        f.write(f"\n# Research Campaign: {config.get('name', 'Unnamed')}\n")
        f.write("## 1. Exploration Phase\n")
        f.write("| Iteration | Action | Fidelity | Reward | Best Reward |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    surrogate = SurrogateModel()
    builder = StructureBuilder()
    compute = ComputeManager(config["compute"] if "compute" in config else config)
    evaluator = EvaluationAgent(governor.get_reward_function(), knowledge_graph)
    
    strategist = OptimizationStrategist(
        surrogate=surrogate, 
        config=config, 
        experiment_db=experiment_db,
        compute_manager=compute,
        builder=builder,
        evaluator=evaluator,
        knowledge_graph=knowledge_graph,
        hypothesis_db=hypothesis_db
    )

    # 2. Initial Hypothesis Formulation (PI Step)
    bulk = config.get("constraints", {}).get("bulk", {})
    keywords = list(bulk.keys())
    claims = literature_db.find_claims(keywords)
    current_hypothesis = pi_agent.formulate_initial_hypothesis(claims, config.get("original_prompt", ""))
    logger.info(f"--- Initial PI Hypothesis: {current_hypothesis.theory_statement} ---")

    # 3. Initial State Setup (Baseline)
    if not experiment_db.get_training_data():
        logger.info("Establishing pristine baseline...")
        current_state = SurfaceState(
            bulk_composition=config["constraints"]["bulk"],
            miller_index=config["constraints"]["facet"],
            termination="default",
            adsorbates=[],
            coverage=0.0
        )
        slab = builder.build_structure(current_state)
        current_state.slab_structure = slab
        
        job_id = compute.submit_job(slab, current_state, sim_type=SimulationType.MLIP, iteration=0)
        results_dir = compute.fetch_results(job_id)
        observables, reward = evaluator.evaluate_calculation(results_dir, {"state": current_state})
        
        init_data = {
            "reward": reward, "total_energy": observables.get("total_energy", 0.0), 
            "coverage": 0.0, "fidelity": observables.get("fidelity", "MLIP")
        }
        experiment_db.add_experiment(current_state, init_data)
        knowledge_graph.record_experiment(current_state, None, init_data)
    
    logger.info("--- CLASDE DISCOVERY LOOP STARTED ---")
    
    # 4. Optimization Loop with Hypothesis Testing
    while governor.has_budget():
        # A. Strategic Observation
        obs = strategist.observe_state()
        strategist.update_belief(obs)
        
        # B. Propose actions influenced by Hypothesis
        candidates = strategist.propose_actions()
        scores = strategist.score_actions(candidates)
        
        import numpy as np
        best_idx = int(np.argmax(scores))
        
        # C. Execute Workflow Graph (influenced by Hypothesis)
        result = strategist.execute_best(candidates[best_idx], hypothesis=current_hypothesis)
        
        while result.get("status") == "pending":
            logger.info(f"Job {result.get('job_id')} pending. Sleeping 60s...")
            time.sleep(60)
            result = strategist.execute_best(candidates[best_idx], hypothesis=current_hypothesis)
            
        governor.consume_budget()
        
        # D. Memory Update
        strategist.update_memory(result)
        metadata = result["metadata"]
        with open(log_file, "a") as f:
            f.write(f"| {metadata['iteration']} | {result['action'].action_type.value} | {metadata['fidelity'].upper()} | {result['reward']:.4f} | {experiment_db.get_best_reward():.4f} |\n")
        
        experiment_db.save()
        knowledge_graph.record_experiment(result["state"], result["action"], result["observables"], metadata)
        
        # E. Hypothesis Verification & Evolution (The Feedback Loop)
        verification_msg = pi_agent.verify_current_hypothesis(current_hypothesis, experiment_db.get_training_data())
        logger.info(f"[PI Verification] {verification_msg}")
        
        current_hypothesis = pi_agent.evolve_hypothesis(current_hypothesis, experiment_db.get_training_data())
        logger.info(f"--- New Hypothesis for Next Iteration: {current_hypothesis.theory_statement} ---")

    # 5. Reasoning & Final Reporting
    patterns = pi_agent.analyze_graph()
    if patterns:
        for p in patterns:
            theory = theory_builder.build_theory(p)
            theory_builder.discovered_laws.append({"type": "custom", "statement": theory})
    
    # Automated physical law discovery
    theory_builder.identify_electronic_descriptors()
    theory_builder.identify_termination_bias()
    
    if config.get("finetune_mlip", False):
        compute.train_chgnet(experiment_db)

    report = theory_builder.generate_report()
    logger.info(f"\n{report}")
    
    # Save the readable Discovery Report to disk
    report_path = os.path.join("data/results", "discovery_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Final Discovery Report saved to {report_path}")

    experiment_db.save()
    hypothesis_db.save()
    literature_db.save()
    kg_memory.save(knowledge_graph)

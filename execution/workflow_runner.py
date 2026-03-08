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
from execution.compute_agent import ComputeManager
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
    """
    Ensures that every scientific campaign is fully traceable and reproducible.
    
    Captures software versions, environment state, and exact random seeds used 
    during the execution of the autonomous loop.
    """
    def capture_environment(self) -> Dict[str, Any]:
        import sys
        import platform
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": time.ctime(),
            "random_seed": 42 # Fixed seed for reproducibility
        }

def run_adsorption_campaign(config: Dict[str, Any]) -> None:
    """
    Orchestrate the high-level CLASDE Bayesian Optimization loop.
    Enhanced with Autonomous Reasoning and Detailed Research Logging.
    """
    repro = ReproducibilityLayer()
    env_metadata = repro.capture_environment()
    
    # Normalize facet to tuple if loaded from YAML
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
    theory_builder = TheoryBuilder(knowledge_graph)
    
    results_dir: str = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    log_file: str = os.path.join(results_dir, "research_log.md")
    
    # Initialize Research Log
    with open(log_file, "a") as f:
        f.write(f"\n# Research Campaign: {config.get('name', 'Unnamed')}\n")
        f.write(f"**Timestamp:** {env_metadata['timestamp']}\n")
        f.write(f"**Reproducibility:** Python {env_metadata['python_version'].split()[0]} on {env_metadata['platform']}\n")
        f.write(f"**Original User Intent:** *\"{config.get('original_prompt', 'N/A')}\"*\n")
        f.write(f"**Scientific Interpretation:** {config.get('description', 'N/A')}\n")
        f.write(f"**Objective Config:** `{config.get('objective')}`\n")
        f.write(f"**Chemistry Constraints:** `{config.get('constraints')}`\n\n")
        f.write("## 1. Exploration Phase\n")
        f.write("| Iteration | Action | Fidelity | Reward | Best Reward |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    surrogate = SurrogateModel()
    builder = StructureBuilder()
    compute = ComputeManager(config["compute"] if "compute" in config else config)
    evaluator = EvaluationAgent(governor.get_reward_function())
    
    # Initialize the fully autonomous Strategist Agent
    strategist = OptimizationStrategist(
        surrogate=surrogate, 
        config=config.get("acquisition", {}),
        experiment_db=experiment_db,
        compute_manager=compute,
        builder=builder,
        evaluator=evaluator,
        knowledge_graph=knowledge_graph,
        hypothesis_db=hypothesis_db
    )

    # 2. Initial State Setup
    if not experiment_db.dataset:
        obj = config.get("objective", {})
        initial_adsorbates = []
        if obj.get("adsorbate"):
            initial_adsorbates.append(AdsorbateInstance(identity=obj["adsorbate"], coverage=0.25))
            
        current_state = SurfaceState(
            bulk_composition=config["constraints"]["bulk"],
            miller_index=config["constraints"]["facet"],
            termination="default",
            adsorbates=initial_adsorbates,
            coverage=sum(a.coverage for a in initial_adsorbates)
        )
        experiment_db.add_experiment(current_state, {"reward": -5.0}) 
    
    logger.info("--- CLASDE ENGINE STARTED ---")
    
    # 3. Main Agentic Optimization Loop
    while governor.has_budget():
        # The Strategist Agent autonomously handles: observe -> update -> propose -> score -> execute -> memory
        result = strategist.run_step()
        governor.consume_budget()
        
        # Log to Research Log
        iteration: int = result["metadata"]["iteration"]
        action_type: str = result["action"].action_type.value
        compute_mode: str = result["metadata"]["fidelity"]
        reward: float = result["reward"]
        best_reward: float = experiment_db.get_best_reward()
        
        with open(log_file, "a") as f:
            f.write(f"| {iteration} | {action_type} | {compute_mode.upper()} | {reward:.4f} | {best_reward:.4f} |\n")

    # 4. Autonomous Reasoning & Hypothesis Evolution
    logger.info("--- PI AGENT REASONING PHASE ---")
    with open(log_file, "a") as f:
        f.write("\n## 2. Scientific Reasoning Phase\n")
        
    patterns = pi_agent.analyze_graph()
    if patterns:
        for p in patterns:
            theory = theory_builder.build_theory(p)
            theory_builder.discovered_laws.append({"type": "custom", "statement": theory})
            logger.info(f"Theory Found: {theory}")
            with open(log_file, "a") as f:
                f.write(f"- **Discovered Theory:** {theory}\n")
        
        new_hypotheses = pi_agent.propose_experiments(patterns)
        with open(log_file, "a") as f:
            f.write(f"\n**PI Agent Recommendation:** Formulated {len(new_hypotheses)} new hypotheses for next-gen campaigns.\n")
    else:
        with open(log_file, "a") as f:
            f.write("- *No statistically significant patterns detected in this budget cycle.*\n")

    # Final Report
    report = theory_builder.generate_report()
    logger.info(f"\n{report}")
    with open(log_file, "a") as f:
        f.write(f"\n### Final Summary\n```text\n{report}\n```\n")
        f.write("-" * 80 + "\n")

    experiment_db.save()
    hypothesis_db.save()
    literature_db.save()
    kg_memory.save(knowledge_graph)
    logger.info("Databases and Knowledge Graph saved to data/results/")
    logger.info(f"Research log updated at {log_file}")

import os
import time
import yaml
from typing import List, Tuple, Dict, Any
from core.state import SurfaceState
from core.action import MutationAction, ActionType
from core.transition import TransitionEngine
from optimization.surrogate import GaussianProcessModel as SurrogateModel
from agents.governor import ResearchGovernor
from agents.strategist import OptimizationStrategist
from agents.builder import StructureBuilder
from agents.compute import ComputeManager
from agents.evaluator import EvaluationAgent
from agents.memory import MemoryGraph
from research.experiment_graph import KnowledgeGraph, ExperimentNode
from research.hypothesis_agent import HypothesisAgent
from research.theory_builder import TheoryBuilder

def run_adsorption_campaign(config: Dict[str, Any]):
    """
    Orchestrate the high-level CLASDE Bayesian Optimization loop.
    Enhanced with Autonomous Reasoning and Detailed Research Logging.
    """
    # Normalize facet to tuple if loaded from YAML
    if "constraints" in config and "facet" in config["constraints"]:
        config["constraints"]["facet"] = tuple(config["constraints"]["facet"])

    # 1. Component Initialization
    governor = ResearchGovernor(config)
    memory = MemoryGraph()
    knowledge_graph = KnowledgeGraph()
    pi_agent = HypothesisAgent(knowledge_graph)
    theory_builder = TheoryBuilder(knowledge_graph)
    
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "clasde_memory.json")
    log_file = os.path.join(results_dir, "research_log.md")
    
    # Initialize Research Log
    with open(log_file, "a") as f:
        f.write(f"\n# Research Campaign: {config.get('name', 'Unnamed')}\n")
        f.write(f"**Timestamp:** {time.ctime()}\n")
        f.write(f"**Original User Intent:** *\"{config.get('original_prompt', 'N/A')}\"*\n")
        f.write(f"**Scientific Interpretation:** {config.get('description', 'N/A')}\n")
        f.write(f"**Objective Config:** `{config.get('objective')}`\n")
        f.write(f"**Chemistry Constraints:** `{config.get('constraints')}`\n\n")
        f.write("## 1. Exploration Phase\n")
        f.write("| Iteration | Action | Fidelity | Reward | Best Reward |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    if os.path.exists(output_file):
        print(f"Loading existing session from {output_file}...")
        memory.load(output_file)
        governor.current_evaluations = len(memory.dataset)

    surrogate = SurrogateModel()
    strategist = OptimizationStrategist(surrogate, config["acquisition"])
    builder = StructureBuilder()
    compute = ComputeManager(config["compute"] if "compute" in config else config)
    evaluator = EvaluationAgent(governor.get_reward_function())
    transition_engine = TransitionEngine()

    # 2. Initial State Setup
    if not memory.dataset:
        current_state = SurfaceState(
            bulk_composition=config["constraints"]["bulk"],
            miller_index=config["constraints"]["facet"],
            termination="default"
        )
        memory.add_state(current_state, reward=-5.0) 
    else:
        last_entry = memory.dataset[-1]
        current_state = last_entry['state']
    
    print(f"--- CLASDE ENGINE STARTED ---")
    
    # 3. Main Optimization Loop
    while governor.has_budget():
        # A. Strategic Decision Phase
        strategist.update_model(memory.get_training_data())
        best_f = memory.get_best_reward()
        action, next_state = strategist.select_next_action(current_state, best_f)
        
        mu, sigma = surrogate.predict(next_state)
        iteration = governor.current_evaluations + 1
        
        # Determine compute mode
        force_mode = config.get("compute", {}).get("mode")
        if force_mode:
            compute_mode = force_mode
        else:
            sigma_threshold = config.get("compute", {}).get("sigma_threshold", 0.5)
            use_vasp = (sigma > sigma_threshold) or (iteration % 5 == 0)
            compute_mode = "vasp" if use_vasp else "local_emt"
        
        print(f"\nIteration {iteration}: {action.action_type}")
        
        # B. Execution Phase
        structure = builder.build_structure(next_state)
        job_id = compute.submit_dft_job(structure, next_state, iteration)
        results_path = compute.fetch_results(job_id)
        
        # C. Evaluation Phase
        observables, reward = evaluator.evaluate_calculation(results_path, {})
        memory.add_state(next_state, observables, reward)
        governor.consume_budget()
        
        # Log to Research Log
        with open(log_file, "a") as f:
            f.write(f"| {iteration} | {action.action_type} | {compute_mode.upper()} | {reward:.4f} | {memory.get_best_reward():.4f} |\n")
        
        # D. Knowledge Graph Update
        exp_node = ExperimentNode(
            node_id=next_state.get_id(),
            state=next_state,
            action=action,
            objective=config["objective"].get("type", "unknown"),
            result={"reward": reward, **observables}
        )
        knowledge_graph.add_experiment(exp_node, parent_id=current_state.get_id())
        current_state = next_state

    # 4. Autonomous Reasoning & Hypothesis Evolution
    print("\n--- PI AGENT REASONING PHASE ---")
    with open(log_file, "a") as f:
        f.write("\n## 2. Scientific Reasoning Phase\n")
        
    patterns = pi_agent.analyze_graph()
    if patterns:
        for p in patterns:
            theory = theory_builder.build_theory(p)
            print(f"  [Theory Found] {theory}")
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
    print(report)
    with open(log_file, "a") as f:
        f.write(f"\n### Final Summary\n```text\n{report}\n```\n")
        f.write("-" * 80 + "\n")

    memory.save(output_file)
    print(f"Memory saved to {output_file}")
    print(f"Research log updated at {log_file}")

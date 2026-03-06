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
    Enhanced with Autonomous Reasoning (PI Agent).
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
    
    # ... rest of initialization ...
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "clasde_memory.json")
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
        # Record initial state baseline
        memory.add_state(current_state, reward=-5.0) 
    else:
        # Resume from the last known state in memory
        last_entry = memory.dataset[-1]
        current_state = last_entry['state']
    
    print(f"--- CLASDE ENGINE STARTED ---")
    print(f"Objective: {config['objective']}")
    
    # 3. Main Optimization Loop
    sigma_threshold = config.get("compute", {}).get("sigma_threshold", 0.5) 
    vasp_frequency = config.get("compute", {}).get("vasp_frequency", 5)
    force_mode = config.get("compute", {}).get("mode")
    
    while governor.has_budget():
        # A. Strategic Decision Phase
        strategist.update_model(memory.get_training_data())
        
        best_f = memory.get_best_reward()
        action, next_state = strategist.select_next_action(current_state, best_f)
        
        # Determine fidelity for this iteration
        mu, sigma = surrogate.predict(next_state)
        
        iteration = governor.current_evaluations + 1
        
        # If mode is explicitly forced in config, use it. Otherwise, use fidelity switching.
        if force_mode:
            compute_mode = force_mode
        else:
            use_vasp = (sigma > sigma_threshold) or (iteration % vasp_frequency == 0)
            compute_mode = "vasp" if use_vasp else "local_emt"
        
        print(f"\nIteration {iteration}:")
        print(f"  Action: {action.action_type} -> {action.parameters}")
        print(f"  Fidelity: {compute_mode.upper()} (sigma={sigma:.3f})")
        
        # Update config temporarily for this job
        compute.config["compute_mode"] = compute_mode
        
        # B. Transition Phase
        memory.add_transition(current_state, action, next_state)
        
        # C. Physical Execution Phase
        structure = builder.build_structure(next_state)
        job_id = compute.submit_dft_job(structure, next_state, iteration)
        
        # D. Evaluation & Reward Phase
        print(f"  Waiting for calculation to complete...")
        
        # In a real HPC scenario, we'd poll monitor_jobs
        results_path = compute.fetch_results(job_id)
        
        # VASP Converge check logic (kept from original loop.py)
        if compute_mode == "vasp":
            max_wait = 3600
            poll_interval = 60
            waited = 0
            while waited < max_wait:
                # This is simplified; in production, use compute.monitor_jobs()
                outcar = os.path.join(results_path, "OUTCAR")
                if os.path.exists(outcar):
                    break
                time.sleep(poll_interval)
                waited += poll_interval
        
        observables, reward = evaluator.evaluate_calculation(results_path, {})
        
        # E. Memory Update
        memory.add_state(next_state, observables, reward)
        governor.consume_budget()
        
        # Populate Knowledge Graph for Hypothesis Agent
        exp_node = ExperimentNode(
            node_id=next_state.get_id(),
            state=next_state,
            action=action,
            objective=config["objective"].get("type", "unknown"),
            result={"reward": reward, **observables}
        )
        knowledge_graph.add_experiment(exp_node, parent_id=current_state.get_id())
        
        # F. State Transition
        current_state = next_state
        print(f"  Observed Reward: {reward:.4f}")
        print(f"  Current Best: {memory.get_best_reward():.4f}")

    print("\n--- CLASDE ENGINE TERMINATED: BUDGET EXHAUSTED ---")
    
    # H. Autonomous Reasoning Phase
    print("\n--- PI AGENT REASONING PHASE ---")
    patterns = pi_agent.analyze_graph()
    if patterns:
        for p in patterns:
            theory = theory_builder.build_theory(p)
            print(f"  [Theory Found] {theory}")
        
        # Propose new campaigns
        new_hypotheses = pi_agent.propose_experiments(patterns)
        print(f"  [PI Agent] Proposed {len(new_hypotheses)} new hypothesis-testing campaigns.")
    else:
        print("  [PI Agent] No statistically significant patterns detected yet.")

    print(theory_builder.generate_report())
    
    # G. Persistence
    memory.save(output_file)
    print(f"Memory graph and dataset saved to {output_file}")

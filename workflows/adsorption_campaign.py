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

def run_adsorption_campaign(config: Dict[str, Any]):
    """
    Orchestrate the high-level CLASDE Bayesian Optimization loop.
    
    This is the standard "slow-but-accurate" discovery loop, typically executed 
    with high-fidelity DFT.
    
    The loop executes the following deterministic sequence:
    1. Update Surrogate: The memory graph trains the GPR model.
    2. Strategize: The Strategist agent scores candidate mutations and selects the best action.
    3. Transition: The logical state is mutated.
    4. Build & Execute: The Builder converts the state to 3D atoms; Compute submits to HPC.
    5. Evaluate: The Evaluator extracts observables and computes the Reward.
    6. Memorize: The graph updates with the new state and reward.
    """
    # Normalize facet to tuple if loaded from YAML
    if "constraints" in config and "facet" in config["constraints"]:
        config["constraints"]["facet"] = tuple(config["constraints"]["facet"])

    # 1. Component Initialization
    governor = ResearchGovernor(config)
    memory = MemoryGraph()
    
    # RESUME LOGIC
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
    sigma_threshold = 0.5 # Trigger VASP if uncertainty > 0.5
    vasp_frequency = 5    # Or every 5 iterations for validation
    
    while governor.has_budget():
        # A. Strategic Decision Phase
        strategist.update_model(memory.get_training_data())
        
        best_f = memory.get_best_reward()
        action, next_state = strategist.select_next_action(current_state, best_f)
        
        # Determine fidelity for this iteration
        mu, sigma = surrogate.predict(next_state)
        
        iteration = governor.current_evaluations + 1
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
        
        # F. State Transition
        current_state = next_state
        print(f"  Observed Reward: {reward:.4f}")
        print(f"  Current Best: {memory.get_best_reward():.4f}")

    print("\n--- CLASDE ENGINE TERMINATED: BUDGET EXHAUSTED ---")
    
    # G. Persistence
    memory.save(output_file)
    print(f"Memory graph and dataset saved to {output_file}")

from typing import List, Tuple, Dict, Any
from core.state import SurfaceState
from core.action import MutationAction, ActionType
from core.transition import TransitionEngine
from core.surrogate import SurrogateModel
from agents.governor import ResearchGovernor
from agents.strategist import OptimizationStrategist
from agents.builder import StructureBuilder
from agents.compute import ComputeManager
from agents.evaluator import EvaluationAgent
from agents.memory import MemoryGraph

def generate_candidates(state: SurfaceState, transition_engine: TransitionEngine) -> List[Tuple[MutationAction, SurfaceState]]:
    """
    Generate next state candidates from the current state using mutation operators.
    In v1, we restrict this to a small set of predefined mutations.
    """
    candidates = []
    
    # 1. Vacancy mutations (enumerated)
    for site in ["O1", "O2"]:
        action = MutationAction(
            action_type=ActionType.INTRODUCE_VACANCY,
            parameters={"site": site, "index": 0}
        )
        candidates.append((action, transition_engine.apply(state, action)))
        
    # 2. Coverage mutations (discrete)
    for new_cov in [0.25, 0.5, 0.75, 1.0]:
        if abs(new_cov - state.coverage) > 0.01:
            action = MutationAction(
                action_type=ActionType.MODIFY_COVERAGE,
                parameters={"coverage": new_cov}
            )
            candidates.append((action, transition_engine.apply(state, action)))
            
    # 3. Termination changes
    for term in ["AO", "BO2"]:
        if term != state.termination:
            action = MutationAction(
                action_type=ActionType.CHANGE_TERMINATION,
                parameters={"termination": term}
            )
            candidates.append((action, transition_engine.apply(state, action)))
            
    return candidates

def main():
    # 0. High-Level Research Objective Configuration
    # (In production, this could be loaded from a YAML or prompted via Governor LLM)
    config = {
        "objective": {
            "type": "adsorption_tuning", 
            "target_e_ads": -1.5
        },
        "budget": {
            "max_evaluations": 10
        },
        "acquisition": {
            "acquisition_type": "EI", 
            "kappa": 2.576
        },
        "constraints": {
            "facet": (0, 0, 1),
            "bulk": {"La": 0.5, "Sr": 0.5, "Mn": 1.0, "O": 3.0}
        }
    }
    
    # 1. Component Initialization
    governor = ResearchGovernor(config)
    memory = MemoryGraph()
    surrogate = SurrogateModel()
    strategist = OptimizationStrategist(surrogate, config["acquisition"])
    builder = StructureBuilder()
    compute = ComputeManager(config)
    evaluator = EvaluationAgent(governor.get_reward_function())
    transition_engine = TransitionEngine()

    # 2. Initial State Setup (The Root Node)
    current_state = SurfaceState(
        bulk_composition=config["constraints"]["bulk"],
        miller_index=config["constraints"]["facet"],
        termination="default"
    )
    
    # Record initial state baseline
    # (Mock evaluation for the first point)
    memory.add_state(current_state, reward=-5.0) 
    
    print(f"--- CLASDE v1 ENGINE STARTED ---")
    print(f"Objective: {config['objective']}")
    
    # 3. Main Optimization Loop
    while governor.has_budget():
        # A. Strategic Decision Phase
        strategist.update_model(memory.get_training_data())
        
        candidates = generate_candidates(current_state, transition_engine)
        best_f = memory.get_best_reward()
        
        action, next_state = strategist.select_next_action(current_state, candidates, best_f)
        
        print(f"\nIteration {governor.current_evaluations + 1}:")
        print(f"  Action: {action.action_type} -> {action.parameters}")
        
        # B. Transition Phase
        memory.add_transition(current_state, action, next_state)
        
        # C. Physical Execution Phase
        structure = builder.build_structure(next_state)
        job_id = compute.submit_dft_job(structure, next_state.get_id())
        
        # D. Evaluation & Reward Phase
        results_path = compute.fetch_results(job_id)
        observables, reward = evaluator.evaluate_calculation(results_path, {})
        
        # E. Memory Update
        memory.add_state(next_state, observables, reward)
        governor.consume_budget()
        
        # F. State Transition
        current_state = next_state
        print(f"  Observed Reward: {reward:.4f}")
        print(f"  Current Best: {memory.get_best_reward():.4f}")

    print("\n--- CLASDE v1 ENGINE TERMINATED: BUDGET EXHAUSTED ---")
    
    # G. Persistence
    output_file = "clasde_memory.json"
    memory.save(output_file)
    print(f"Memory graph and dataset saved to {output_file}")

if __name__ == "__main__":
    main()

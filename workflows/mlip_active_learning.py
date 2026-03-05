import os
import yaml
import time
from typing import Dict, Any
from core.state import SurfaceState
from agents.mlip_manager import MLIPManager
from agents.dynamics import DynamicsAgent
from agents.builder import StructureBuilder
from agents.compute import ComputeManager
from agents.evaluator import EvaluationAgent
from agents.memory import MemoryGraph
from agents.governor import ResearchGovernor

def run_active_learning_loop(config: Dict[str, Any]):
    """
    Orchestrate the MLIP-driven Active Learning Loop.
    
    This is the "Self-Driving Lab" mode. It bypasses the slow BO loop by using a 
    Machine Learning Interatomic Potential (MLIP) to rapidly explore the configuration 
    space via Molecular Dynamics or Structural Relaxation.
    
    The loop executes the following sequence:
    1. Explore (MLIP): Run MD/Relaxation using the fast, surrogate potential.
    2. Detect Uncertainty: Query the MLIP for its confidence (sigma) on the new structure.
    3. Correct (DFT): If the MLIP is highly uncertain (out-of-distribution), trigger 
       a high-fidelity DFT calculation.
    4. Retrain: Add the new DFT data to the MLIP training set and refit.
    
    This allows 24/7 exploration, only consuming expensive HPC time when encountering 
    novel physical phenomena.
    """
    # 1. Initialization
    mlip = MLIPManager()
    dynamics = DynamicsAgent(mlip)
    builder = StructureBuilder()
    compute = ComputeManager(config)
    evaluator = EvaluationAgent(None) # Reward handled differently here
    memory = MemoryGraph()
    governor = ResearchGovernor(config)
    
    current_state = SurfaceState(
        bulk_composition=config["constraints"]["bulk"],
        miller_index=config["constraints"]["facet"],
        termination="default"
    )
    
    sigma_threshold = config.get("active_learning", {}).get("sigma_threshold", 0.1)
    
    print("--- CLASDE ACTIVE LEARNING LOOP STARTED ---")
    
    while governor.has_budget():
        iteration = governor.current_evaluations + 1
        print(f"\nIteration {iteration}:")
        
        # A. MLIP Exploration Phase
        # Run MD or relaxation to find new stable/interesting configurations
        structure = builder.build_structure(current_state)
        relaxed_structure, _ = dynamics.relax(structure)
        
        # B. Uncertainty Detection Phase
        energy_mlip, sigma = mlip.predict_energy(relaxed_structure)
        print(f"  MLIP Prediction: E={energy_mlip:.4f}, sigma={sigma:.4f}")
        
        need_dft = (sigma > sigma_threshold) or (iteration == 1)
        
        if need_dft:
            print("  [!] High Uncertainty: Triggering DFT Correction.")
            # C. DFT Correction Phase
            job_id = compute.submit_dft_job(relaxed_structure, current_state, iteration)
            results_path = compute.fetch_results(job_id)
            observables, reward = evaluator.evaluate_calculation(results_path, {})
            
            dft_energy = observables.get("total_energy", 0.0)
            print(f"  DFT Energy: {dft_energy:.4f}")
            
            # D. Retraining Phase
            mlip.add_data(relaxed_structure, dft_energy)
            mlip.train()
            
            # E. Memory & State Update
            memory.add_state(current_state, observables, reward)
            governor.consume_budget()
        else:
            print("  [✓] MLIP Confidence High: Skipping DFT.")
            # Move to next state based on MLIP relaxation
            # (In a real scenario, we'd update current_state from relaxed_structure)
        
        # Simplified: Mutation for next iteration
        # In production, this would use the Strategist
        time.sleep(1) 

    print("\n--- ACTIVE LEARNING CAMPAIGN FINISHED ---")

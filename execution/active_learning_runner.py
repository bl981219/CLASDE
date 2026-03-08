import logging
import os
import yaml
import time
from typing import Dict, Any, List, Optional

from core.state import SurfaceState
from execution.mlip_manager import MLIPManager
from execution.dynamics_engine import DynamicsAgent
from agents.builder_agent import StructureBuilder
from execution.compute_agent import ComputeManager, SimulationType
from agents.evaluator_agent import EvaluationAgent
from memory.experiment_db import ExperimentDatabase
from agents.governor_agent import ResearchGovernor

# Configure logger
logger = logging.getLogger(__name__)

def run_active_learning_loop(config: Dict[str, Any]) -> None:
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
    """
    # 1. Initialization
    mlip = MLIPManager()
    dynamics = DynamicsAgent(mlip)
    builder = StructureBuilder()
    compute = ComputeManager(config)
    evaluator = EvaluationAgent(None) # Reward handled differently here
    experiment_db = ExperimentDatabase()
    governor = ResearchGovernor(config)
    
    current_state = SurfaceState(
        bulk_composition=config["constraints"]["bulk"],
        miller_index=config["constraints"]["facet"],
        termination="default"
    )
    
    sigma_threshold: float = config.get("active_learning", {}).get("sigma_threshold", 0.1)
    
    logger.info("--- CLASDE ACTIVE LEARNING LOOP STARTED ---")
    
    while governor.has_budget():
        iteration = governor.current_evaluations + 1
        logger.info(f"Iteration {iteration}:")
        
        # A. MLIP Exploration Phase
        # Run MD or relaxation to find new stable/interesting configurations
        structure = builder.build_structure(current_state)
        relaxed_structure, _ = dynamics.relax(structure)
        
        # B. Uncertainty Detection Phase
        energy_mlip, sigma = mlip.predict_energy(relaxed_structure)
        logger.info(f"MLIP Prediction: E={energy_mlip:.4f}, sigma={sigma:.4f}")
        
        need_dft = (sigma > sigma_threshold) or (iteration == 1)
        
        if need_dft:
            logger.info("High Uncertainty: Triggering DFT Correction.")
            # C. DFT Correction Phase
            job_id = compute.submit_job(relaxed_structure, current_state, sim_type=SimulationType.DFT, iteration=iteration)
            results_path = compute.fetch_results(job_id)
            observables, reward = evaluator.evaluate_calculation(results_path, {})
            
            dft_energy = observables.get("total_energy", 0.0)
            logger.info(f"DFT Energy: {dft_energy:.4f}")
            
            # D. Retraining Phase
            mlip.add_data(relaxed_structure, dft_energy)
            mlip.train()
            
            # E. Memory & State Update
            experiment_db.add_experiment(current_state, observables)
            governor.consume_budget()
        else:
            logger.info("MLIP Confidence High: Skipping DFT.")
        
        # Simplified: Mutation for next iteration
        time.sleep(1) 

    logger.info("--- ACTIVE LEARNING CAMPAIGN FINISHED ---")

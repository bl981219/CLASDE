import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from optimization.surrogate_models import SurrogateModel
from optimization.acquisition_functions import (
    AcquisitionFunction, 
    ExpectedImprovement, 
    UpperConfidenceBound, 
    ThompsonSampling,
    ScientificDiscoveryAcquisition
)
from optimization.campaign_optimizer import CampaignOptimizer
from core.state import SurfaceState
from core.action import MutationAction, ActionType
from core.transition import TransitionEngine
from agents.base_agent import BaseAgent
from execution.compute_agent import SimulationType
from agents.planner_agent import ResearchPlanner, WorkflowTask

logger = logging.getLogger(__name__)

class ActionProposer:
    """
    Component responsible for suggesting candidate mutation actions.
    
    This acts as the "idea generator." It scans the current state and enumerates 
    physically permissible mutations (e.g., adding a vacancy, changing coverage).
    It separates the logic of *proposing* an experiment from the logic of *selecting* it.
    This architecture natively supports LLM-driven heuristic proposals in the future.
    """
    def propose_actions(self, state: SurfaceState) -> List[MutationAction]:
        """
        Suggest mutation operators for the current state.
        This is where 'Expert Knowledge' or 'LLM Heuristics' are applied.
        """
        actions: List[MutationAction] = []
        
        # 1. Heuristic: Vacancies for all elements in bulk
        for el in state.bulk_composition.keys():
            actions.append(MutationAction(
                action_type=ActionType.INTRODUCE_VACANCY,
                parameters={"site": el, "index": 0}
            ))
            
        # 2. Heuristic: Coverage modifications
        for new_cov in [0.25, 0.5, 0.75, 1.0]:
            if abs(new_cov - state.coverage) > 0.01:
                actions.append(MutationAction(
                    action_type=ActionType.MODIFY_COVERAGE,
                    parameters={"coverage": new_cov}
                ))
                
        # 3. Heuristic: Substitutions
        if "Mn" in state.bulk_composition:
            actions.append(MutationAction(
                action_type=ActionType.SUBSTITUTIONAL_DOPANT,
                parameters={"original_element": "Mn", "dopant": "Sr"}
            ))

        # 4. Heuristic: Swapping (Segregation)
        if "Sr" in state.bulk_composition and "La" in state.bulk_composition:
            actions.append(MutationAction(
                action_type=ActionType.SWAP_ATOMS,
                parameters={"element_a": "La", "element_b": "Sr", "direction": "surface_to_bulk"}
            ))
            actions.append(MutationAction(
                action_type=ActionType.SWAP_ATOMS,
                parameters={"element_a": "Sr", "element_b": "La", "direction": "surface_to_bulk"}
            ))
            
        return actions

class OptimizationStrategist(BaseAgent):
    """
    Agent 2 — Optimization Strategist (The Senior Postdoc).
    
    A true autonomous agent implementing the observe -> update_belief -> 
    propose -> score -> execute -> update_memory lifecycle.
    
    It maintains a belief state (the SurrogateModel), evaluates uncertainty 
    via AcquisitionFunctions, and decides which physical experiment to run next.
    """
    def __init__(self, surrogate: SurrogateModel, config: Dict[str, Any], 
                 experiment_db: Any, compute_manager: Any, builder: Any, evaluator: Any, 
                 knowledge_graph: Any, hypothesis_db: Any, proposer: Optional[ActionProposer] = None) -> None:
        super().__init__()
        self.config = config
        self.proposer = proposer or ActionProposer()
        self.transition_engine = TransitionEngine()
        
        # External Tools & Interfaces
        self.experiment_db = experiment_db
        self.knowledge_graph = knowledge_graph
        self.compute = compute_manager
        self.builder = builder
        self.evaluator = evaluator
        
        # Internal Agents
        self.planner = ResearchPlanner(knowledge_graph, experiment_db, hypothesis_db)
        
        # Belief State Initialization
        self.belief_state = surrogate
        acq_type = self.config.get("acquisition_type", "EI")
        acq_func: AcquisitionFunction
        if acq_type == "EI":
            acq_func = ExpectedImprovement(best_observed_f=-1e9)
        elif acq_type == "UCB":
            acq_func = UpperConfidenceBound(kappa=self.config.get("kappa", 2.576))
        elif acq_type == "SCIENTIFIC":
            acq_func = ScientificDiscoveryAcquisition(
                beta=self.config.get("beta", 1.0),
                gamma=self.config.get("gamma", 0.5),
                delta=self.config.get("delta", 0.1)
            )
        elif acq_type == "TS":
            acq_func = ThompsonSampling()
        else:
            raise ValueError(f"Unknown acquisition type: {acq_type}")
            
        self.optimizer = CampaignOptimizer(self.belief_state, acq_func)
        self.current_state: Optional[SurfaceState] = None
        self.iteration = 0

    def observe_state(self) -> List[Dict[str, Any]]:
        """Observe the current dataset from ExperimentDB."""
        if not self.experiment_db.dataset:
            raise ValueError("Cannot observe empty memory. Seed state required.")
        self.current_state = self.experiment_db.dataset[-1]['state']
        return self.experiment_db.get_training_data()

    def update_belief(self, observations: List[Dict[str, Any]]) -> None:
        """Update the surrogate model based on historical observations."""
        self.optimizer.update(observations)
        if isinstance(self.optimizer.acquisition, ExpectedImprovement):
            self.optimizer.acquisition.best_observed_f = self.experiment_db.get_best_reward()

    def propose_actions(self) -> List[Tuple[MutationAction, SurfaceState]]:
        """Generate candidate mutations and project them into new states."""
        if self.current_state is None:
            raise ValueError("Current state is not set.")
        actions = self.proposer.propose_actions(self.current_state)
        candidates: List[Tuple[MutationAction, SurfaceState]] = []
        for action in actions:
            next_state = self.transition_engine.apply(self.current_state, action)
            candidates.append((action, next_state))
        if not candidates:
            raise ValueError("No candidate actions generated by proposer.")
        return candidates

    def score_actions(self, candidates: List[Tuple[MutationAction, SurfaceState]]) -> List[float]:
        """Evaluate candidates using the belief state, weighing reward vs uncertainty and novelty."""
        # Prepare context for scientific discovery
        existing_feats = [entry['state'].feature_vector for entry in self.experiment_db.dataset]
        context: Dict[str, Any] = {"existing_features": existing_feats}
        
        scores: List[float] = []
        for action, state in candidates:
            # Prepare context for this specific action-state pair
            item_context = context.copy()
            item_context["action"] = action
            
            score = self.optimizer.acquisition.compute_score(state, self.belief_state, context=item_context)
            scores.append(score)
        return scores

    def execute_best(self, best_action_tuple: Tuple[MutationAction, SurfaceState]) -> Dict[str, Any]:
        """Commit to an action, interacting with dynamic Planner and Compute environments."""
        action, next_state = best_action_tuple
        self.iteration += 1
        
        # 1. Dynamic Workflow Generation: Ask the Planner what to do with this state
        workflow_sequence = self.planner.plan_next_steps(next_state)
        
        # 2. Uncertainty Reasoning: Decide fidelity based on model confidence
        mu, sigma = self.belief_state.predict(next_state)
        sigma_threshold = self.config.get("compute", {}).get("sigma_threshold", 0.5)
        use_vasp = (sigma > sigma_threshold) or (self.iteration % 5 == 0)
        sim_type = SimulationType.DFT if use_vasp else SimulationType.MLIP
        
        # 3. Execution of the Sequence (Simplified for demo)
        logger.info(f"Iteration {self.iteration}: {action.action_type}")
        logger.info(f"  [Planner] Sequence: {' -> '.join([t.value for t in workflow_sequence])}")
        
        structure = self.builder.build_structure(next_state)
        job_id = self.compute.submit_job(structure, next_state, sim_type=sim_type, iteration=self.iteration)
        results_path = self.compute.fetch_results(job_id)
        
        observables, reward = self.evaluator.evaluate_calculation(results_path, {})
        
        return {
            "state": next_state,
            "action": action,
            "reward": reward,
            "observables": observables,
            "metadata": {
                "iteration": self.iteration, 
                "fidelity": sim_type.value, 
                "sigma": float(sigma),
                "workflow": [t.value for t in workflow_sequence]
            }
        }

    def update_memory(self, result: Dict[str, Any]) -> None:
        """Update both the local trajectory and the global semantic Knowledge Graph."""
        next_state = result["state"]
        reward = result["reward"]
        observables = result["observables"]
        action = result["action"]
        metadata = result["metadata"]

        # Add to centralized experiment database
        self.experiment_db.add_experiment(
            state=next_state,
            results={**observables, "reward": reward, **metadata},
            action=action,
            parent_state=self.current_state
        )

        # Semantic record in Knowledge Graph

        self.knowledge_graph.record_experiment(
            state=next_state,
            action=action,
            result_data={"reward": reward, **observables},
            calc_metadata=metadata
        )
        
        logger.info(f"  Observed Reward: {reward:.4f}")
        logger.info(f"  Current Best: {self.experiment_db.get_best_reward():.4f}")

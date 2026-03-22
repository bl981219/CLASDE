import logging
import os
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
from core.campaign import ResearchMode
from execution.compute_agent import SimulationType, JobStatus
from agents.planner_agent import ResearchPlanner
from core.workflow_graph import WorkflowExecutor

logger = logging.getLogger(__name__)

class ActionProposer:
    """
    Component responsible for suggesting candidate mutation actions.
    """
    def __init__(self, allowed_actions: Optional[List[str]] = None) -> None:
        self.allowed_actions = allowed_actions

    def propose_actions(self, state: SurfaceState) -> List[MutationAction]:
        """Suggest mutation operators for the current state with high diversity."""
        all_possible: List[MutationAction] = []
        bulk_elements = [el for el in state.bulk_composition.keys() if el != "O"]
        
        # 1. Surface Vacancies
        for el in bulk_elements + ["O"]:
            for depth in [0, 1]:
                all_possible.append(MutationAction(
                    action_type=ActionType.INTRODUCE_VACANCY,
                    parameters={"site": el, "index": depth}
                ))
            
        # 2. Cation Swapping (Surface <-> Subsurface)
        if len(bulk_elements) > 1:
            for i in range(len(bulk_elements)):
                for j in range(i + 1, len(bulk_elements)):
                    for da, db in [(0, 1), (0, 2)]:
                        all_possible.append(MutationAction(
                            action_type=ActionType.SWAP_ATOMS,
                            parameters={
                                "element_a": bulk_elements[i], 
                                "element_b": bulk_elements[j],
                                "depth_a": da, "depth_b": db
                            }
                        ))

        # 3. Substitutional Doping
        from ase.data import atomic_numbers
        for el in bulk_elements:
            z = atomic_numbers.get(el)
            if z:
                for shift in [-1, 1]:
                    dopant = next((s for s, n in atomic_numbers.items() if n == z + shift), None)
                    if dopant:
                        all_possible.append(MutationAction(
                            action_type=ActionType.SUBSTITUTIONAL_DOPANT,
                            parameters={"original_element": el, "dopant": dopant}
                        ))
        
        # 4. Environment/Condition Changes
        all_possible.append(MutationAction(
            action_type=ActionType.MODIFY_ENVIRONMENT,
            parameters={"temperature": 800} # K
        ))
        
        return all_possible

class OptimizationStrategist(BaseAgent):
    """
    Agent 2 — Optimization Strategist (The Senior Postdoc).
    """
    def __init__(self, surrogate: SurrogateModel, config: Dict[str, Any], 
                 experiment_db: Any, compute_manager: Any, builder: Any, evaluator: Any, 
                 knowledge_graph: Any, hypothesis_db: Any, proposer: Optional[ActionProposer] = None) -> None:
        super().__init__()
        self.config = config
        self.proposer = proposer or ActionProposer(allowed_actions=config.get("allowed_actions"))
        self.transition_engine = TransitionEngine()
        self.experiment_db = experiment_db
        self.knowledge_graph = knowledge_graph
        self.compute = compute_manager
        self.builder = builder
        self.evaluator = evaluator
        self.planner = ResearchPlanner(knowledge_graph, experiment_db, hypothesis_db)
        
        # Acquisition
        acq_type = self.config.get("acquisition_type", "EI")
        if acq_type == "EI":
            acq_func = ExpectedImprovement(best_observed_f=self.experiment_db.get_best_reward())
        elif acq_type == "UCB":
            acq_func = UpperConfidenceBound(kappa=self.config.get("kappa", 2.576))
        else:
            acq_func = ThompsonSampling()
            
        self.optimizer = CampaignOptimizer(surrogate, acq_func)
        self.executor = WorkflowExecutor(self)
        self.belief_state = surrogate
        self.current_state: Optional[SurfaceState] = None
        self.pending_state: Optional[SurfaceState] = None
        self.iteration = 0

    def observe_state(self) -> List[Dict[str, Any]]:
        training_data = self.experiment_db.get_training_data()
        if not training_data:
            raise ValueError("No data in ExperimentDB.")
        self.current_state = training_data[-1]['state']
        return training_data

    def update_belief(self, observations: List[Dict[str, Any]]) -> None:
        self.optimizer.update(observations)

    def propose_actions(self) -> List[Tuple[MutationAction, SurfaceState]]:
        """
        Generate candidate mutations and project them into new states.
        Ensures candidates are unique and haven't been evaluated yet.
        """
        if self.current_state is None:
            raise ValueError("Current state is not set.")
            
        actions = self.proposer.propose_actions(self.current_state)
        candidates: List[Tuple[MutationAction, SurfaceState]] = []
        
        # Track seen IDs to ensure diversity
        seen_ids = {self.current_state.get_id()}
        for entry in self.experiment_db.get_training_data():
            seen_ids.add(entry['state'].get_id())

        for action in actions:
            try:
                next_state = self.transition_engine.apply(self.current_state, action)
                next_id = next_state.get_id()
                if next_id not in seen_ids:
                    candidates.append((action, next_state))
                    seen_ids.add(next_id)
            except:
                continue
                
        return candidates

    def score_actions(self, candidates: List[Tuple[MutationAction, SurfaceState]]) -> List[float]:
        from science.validator import DomainValidator
        existing_feats = [entry['state'].get_feature_vector() for entry in self.experiment_db.get_training_data()]
        scores = []
        for action, state in candidates:
            score = self.optimizer.acquisition.compute_score(state, self.belief_state, context={"existing_features": existing_feats, "action": action})
            is_neutral, _ = DomainValidator.validate_charge_neutrality(state.bulk_composition)
            if not is_neutral: score -= 5.0
            scores.append(score)
        return scores

    def execute_best(self, best_action_tuple: Tuple[MutationAction, SurfaceState], hypothesis: Optional[Any] = None) -> Dict[str, Any]:
        action, next_state = best_action_tuple
        self.pending_state = next_state
        self.iteration += 1
        
        workflow_graph = self.planner.plan_next_steps(next_state, hypothesis=hypothesis)
        mu, sigma = self.belief_state.predict(next_state)
        
        compute_config = self.config.get("compute", {})
        force_mode = compute_config.get("mode", "mixed")
        
        if force_mode in ["vasp", "dft"]: use_vasp = True
        elif force_mode in ["chgnet", "local_emt"]: use_vasp = False
        else:
            sigma_thresh = self.config.get("acquisition", {}).get("sigma_threshold", 0.5)
            use_vasp = (sigma > sigma_thresh) or (self.iteration % 5 == 0)

        sim_type = SimulationType.DFT if use_vasp else SimulationType.MLIP
        task_result = self.executor.execute(workflow_graph, sim_type, self.iteration)
        
        if isinstance(task_result, dict):
            result = task_result
        else:
            # Unwrap TaskResult
            result = {
                "status": task_result.status,
                **task_result.result_data,
                "metadata": {
                    "workflow": workflow_graph.name,
                    "sigma": float(sigma),
                    **task_result.metadata
                }
            }

        result["action"] = action
        return result

    def update_memory(self, result: Dict[str, Any]) -> None:
        next_state = result["state"]
        reward = result["reward"]
        observables = result["observables"]
        action = result["action"]
        metadata = result["metadata"]

        self.experiment_db.add_experiment(state=next_state, results={**observables, "reward": reward, **metadata}, action=action, parent_state=self.current_state)
        self.knowledge_graph.record_experiment(state=next_state, action=action, result_data={"reward": reward, **observables}, calc_metadata=metadata)
        
        logger.info(f"  Observed Reward: {reward:.4f}")
        logger.info(f"  Current Best: {self.experiment_db.get_best_reward():.4f}")

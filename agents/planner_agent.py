import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from core.state import SurfaceState
from core.action import MutationAction
import networkx as nx

logger = logging.getLogger(__name__)

class WorkflowTask(str, Enum):
    RELAX = "relax"
    MD = "molecular_dynamics"
    NEB = "nudged_elastic_band"
    ADSORPTION = "adsorption_energy"
    DOS = "density_of_states"
    MLIP_TRAIN = "mlip_training"

class ResearchPlanner(BaseAgent):
    """
    Agent 0.5 — The Research Planner.
    
    Instead of hardcoded pipelines, this agent dynamically constructs 
    task sequences (directed acyclic graphs) based on scientific necessity.
    
    Example: 
    If a surface is highly unstable, it might decide to run MD first 
    to find a local minimum before attempting a high-fidelity relaxation.
    """
    def __init__(self, knowledge_graph: Any, experiment_db: Any, hypothesis_db: Any) -> None:
        super().__init__()
        self.kg = knowledge_graph
        self.exp_db = experiment_db
        self.hyp_db = hypothesis_db
        
        # Belief state: A map of (state, task) -> estimated_utility
        self.belief_state: Dict[Tuple[Any, WorkflowTask], float] = {} 

    def observe_state(self) -> Dict[str, Any]:
        """Observe current system state and recent failures/successes."""
        recent_experiments = self.exp_db.dataset[-5:] if self.exp_db.dataset else []
        top_hypotheses = self.hyp_db.get_top_hypotheses(limit=3)
        return {
            "recent": recent_experiments,
            "top_theories": top_hypotheses
        }

    def update_belief(self, observations: Dict[str, Any]) -> None:
        """Update utility estimates for different task types."""
        # Heuristic: If recent jobs failed convergence, increase utility of MD (pre-equilibration)
        # If uncertainty is high, increase utility of MLIP_TRAIN
        pass

    def propose_actions(self) -> List[List[WorkflowTask]]:
        """Propose candidate workflow sequences."""
        # Simple sequences for demonstration
        candidates = [
            [WorkflowTask.RELAX, WorkflowTask.ADSORPTION],
            [WorkflowTask.MD, WorkflowTask.RELAX, WorkflowTask.ADSORPTION],
            [WorkflowTask.RELAX, WorkflowTask.DOS, WorkflowTask.ADSORPTION],
            [WorkflowTask.MLIP_TRAIN, WorkflowTask.RELAX]
        ]
        return candidates

    def score_actions(self, candidates: List[List[WorkflowTask]]) -> List[float]:
        """Score workflows based on scientific goal and resource cost."""
        scores: List[float] = []
        for sequence in candidates:
            # Heuristic score based on length and task diversity
            score = 1.0 / len(sequence) 
            if WorkflowTask.MD in sequence:
                score += 0.5 # Bias towards stability checks
            scores.append(score)
        return scores

    def execute_best(self, best_sequence: List[WorkflowTask]) -> List[WorkflowTask]:
        """Commit to a dynamic workflow sequence."""
        logger.info(f"[Planner] Dynamically generated workflow: {' -> '.join([t.value for t in best_sequence])}")
        return best_sequence

    def update_memory(self, result: List[WorkflowTask]) -> None:
        """Record the planned sequence in the knowledge graph."""
        pass

    def plan_next_steps(self, state: SurfaceState) -> List[WorkflowTask]:
        """Convenience method to get a sequence for a given state."""
        return self.run_step()

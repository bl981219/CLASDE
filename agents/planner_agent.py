import logging
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from core.state import SurfaceState
from core.action import MutationAction
from core.workflow_graph import WorkflowGraph, TaskNode, TaskType

logger = logging.getLogger(__name__)

class ResearchPlanner(BaseAgent):
    """
    Agent 0.5 — The Research Planner.
    
    Dynamically constructs task sequences (directed acyclic graphs) 
    based on scientific necessity.
    """
    def __init__(self, knowledge_graph: Any, experiment_db: Any, hypothesis_db: Any) -> None:
        super().__init__()
        self.kg = knowledge_graph
        self.exp_db = experiment_db
        self.hyp_db = hypothesis_db
        
        # Belief state: A map of context -> boolean
        self.belief_state: Dict[str, bool] = {} 

    def observe_state(self) -> Dict[str, Any]:
        """Observe current system state and recent failures/successes."""
        recent_experiments = self.exp_db.get_training_data()[-5:] if self.exp_db.get_training_data() else []
        top_hypotheses = self.hyp_db.hypotheses[:3] # Simplified access
        return {
            "recent": recent_experiments,
            "top_theories": top_hypotheses
        }

    def update_belief(self, observations: Dict[str, Any]) -> None:
        """Update utility estimates for different task types."""
        recent_failed = False
        high_uncertainty = False
        
        for exp in observations.get("recent", []):
            if exp.get("status") == "failed":
                recent_failed = True
            if exp.get("metadata", {}).get("sigma", 0) > 0.5:
                high_uncertainty = True
                
        self.belief_state["needs_md"] = recent_failed
        self.belief_state["needs_dos"] = True # We almost always want DOS for analysis
        self.belief_state["needs_neb"] = len(observations.get("top_theories", [])) > 0

    def propose_actions(self) -> List[WorkflowGraph]:
        """Propose candidate workflow DAGs."""
        candidates = []
        
        # Candidate 1: Standard Adsorption Workflow
        g1 = WorkflowGraph("Standard Adsorption")
        b1 = TaskNode(task_type=TaskType.BUILD_SLAB)
        r1 = TaskNode(task_type=TaskType.RELAX_SLAB)
        e1 = TaskNode(task_type=TaskType.ENUMERATE_SITES)
        a1 = TaskNode(task_type=TaskType.CALC_ADSORPTION)
        
        b1_id = g1.add_task(b1)
        r1_id = g1.add_task(r1)
        e1_id = g1.add_task(e1)
        a1_id = g1.add_task(a1)
        
        g1.add_dependency(b1_id, r1_id)
        g1.add_dependency(r1_id, e1_id)
        g1.add_dependency(e1_id, a1_id)
        candidates.append(g1)
        
        # Candidate 2: Unstable Surface Workflow (includes MD)
        g2 = WorkflowGraph("Unstable Surface Recovery")
        b2 = TaskNode(task_type=TaskType.BUILD_SLAB)
        m2 = TaskNode(task_type=TaskType.RUN_MD)
        r2 = TaskNode(task_type=TaskType.RELAX_SLAB)
        e2 = TaskNode(task_type=TaskType.ENUMERATE_SITES)
        a2 = TaskNode(task_type=TaskType.CALC_ADSORPTION)
        
        b2_id = g2.add_task(b2)
        m2_id = g2.add_task(m2)
        r2_id = g2.add_task(r2)
        e2_id = g2.add_task(e2)
        a2_id = g2.add_task(a2)
        
        g2.add_dependency(b2_id, m2_id)
        g2.add_dependency(m2_id, r2_id)
        g2.add_dependency(r2_id, e2_id)
        g2.add_dependency(e2_id, a2_id)
        candidates.append(g2)

        return candidates

    def score_actions(self, candidates: List[WorkflowGraph]) -> List[float]:
        """Score workflows based on scientific goal and current belief state."""
        scores: List[float] = []
        for graph in candidates:
            score = 0.0
            task_types = [t.task_type for t in graph.nodes.values()]
            
            if self.belief_state.get("needs_md") and TaskType.RUN_MD in task_types:
                score += 2.0
            
            # Default to standard workflow if no specific needs
            if score == 0.0 and TaskType.RUN_MD not in task_types:
                score = 1.0
                
            scores.append(score)
        return scores

    def execute_best(self, best_graph: WorkflowGraph) -> WorkflowGraph:
        """Commit to a dynamic workflow graph."""
        logger.info(f"[Planner] Dynamically generated workflow DAG: {best_graph.name}")
        return best_graph

    def update_memory(self, result: WorkflowGraph) -> None:
        """Record the planned sequence in the knowledge graph."""
        pass

    def plan_next_steps(self, state: SurfaceState) -> WorkflowGraph:
        """Convenience method to get a sequence for a given state."""
        return self.run_step()

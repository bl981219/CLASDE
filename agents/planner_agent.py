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
        self.belief_state: Dict[str, bool] = {} 
        self.current_state: Optional[SurfaceState] = None
        self.current_hypothesis: Optional[Any] = None

    def observe_state(self) -> Dict[str, Any]:
        """Observe current system state and recent history."""
        return {
            "recent": self.exp_db.get_training_data()[-5:] if self.exp_db.get_training_data() else [],
            "state": self.current_state,
            "hypothesis": self.current_hypothesis
        }

    def update_belief(self, observations: Dict[str, Any]) -> None:
        """Update utility estimates for different task types."""
        recent_failed = any(exp.get("status") == "failed" for exp in observations.get("recent", []))
        self.belief_state["needs_md"] = recent_failed
        
        hyp = observations.get("hypothesis")
        if hyp:
            theory = hyp.theory_statement.lower()
            if "stability" in theory or "unstable" in theory:
                self.belief_state["needs_md"] = True

    def propose_actions(self) -> List[WorkflowGraph]:
        """Propose candidate workflow graphs."""
        if not self.current_state: return []
        # In this simplified agent, we just return one dynamically composed graph
        return [self.plan_next_steps(self.current_state, self.current_hypothesis)]

    def score_actions(self, actions: List[WorkflowGraph]) -> List[float]:
        """Score candidate graphs."""
        return [1.0 for _ in actions]

    def execute_best(self, best_graph: WorkflowGraph) -> WorkflowGraph:
        """Commit to a workflow."""
        logger.info(f"[Planner] Committing to workflow: {best_graph.name}")
        return best_graph

    def update_memory(self, result: WorkflowGraph) -> None:
        """Record the plan."""
        pass

    def plan_next_steps(self, state: SurfaceState, hypothesis: Optional[Any] = None) -> WorkflowGraph:
        """
        Dynamically composes a WorkflowGraph based on the state and hypothesis.
        Ensures strict validation of task completeness for the campaign objective.
        """
        self.current_state = state
        self.current_hypothesis = hypothesis
        
        graph_name = f"Plan for {state.get_id()[:8]}"
        graph = WorkflowGraph(graph_name)
        
        # 1. Core Tasks (Build and Relax are mandatory)
        build_node = TaskNode(task_type=TaskType.BUILD_SLAB)
        relax_node = TaskNode(task_type=TaskType.RELAX_SLAB)
        
        b_id = graph.add_task(build_node)
        r_id = graph.add_task(relax_node)
        graph.add_dependency(b_id, r_id)
        
        last_node_id = r_id
        
        # 2. Hypothesis-Driven Tasks (Dynamic Expansion)
        if hypothesis:
            theory = hypothesis.theory_statement.lower()
            if any(k in theory for k in ["stability", "dynamic", "unstable", "reconstruction"]) or self.belief_state.get("needs_md"):
                md_node = TaskNode(task_type=TaskType.RUN_MD)
                m_id = graph.add_task(md_node)
                graph.add_dependency(b_id, m_id)
                graph.add_dependency(m_id, r_id)

            if any(k in theory for k in ["electronic", "mechanism", "bonding", "descriptor"]):
                dos_node = TaskNode(task_type=TaskType.CALC_DOS)
                d_id = graph.add_task(dos_node)
                graph.add_dependency(r_id, d_id)
                
            if any(k in theory for k in ["barrier", "kinetic", "diffusion", "transition"]):
                neb_node = TaskNode(task_type=TaskType.RUN_NEB)
                n_id = graph.add_task(neb_node)
                graph.add_dependency(r_id, n_id)

        # 3. Goal-Driven Validation (STRICT RULES)
        if state.adsorbates:
            site_node = TaskNode(task_type=TaskType.ENUMERATE_SITES)
            ads_node = TaskNode(task_type=TaskType.CALC_ADSORPTION)
            s_id = graph.add_task(site_node)
            a_id = graph.add_task(ads_node)
            graph.add_dependency(last_node_id, s_id)
            graph.add_dependency(s_id, a_id)
            last_node_id = a_id
        
        if any(d["type"] == "substitution" for d in state.defects):
            dos_node = TaskNode(task_type=TaskType.CALC_DOS)
            if dos_node.task_type not in [t.task_type for t in graph.nodes.values()]:
                d_id = graph.add_task(dos_node)
                graph.add_dependency(r_id, d_id)

        logger.info(f"[Planner] Dynamically composed workflow: {graph.summarize()}")
        return graph

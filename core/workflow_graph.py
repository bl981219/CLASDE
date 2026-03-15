import logging
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
import networkx as nx
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    BUILD_SLAB = "build_slab"
    RELAX_SLAB = "relax_slab"
    ENUMERATE_SITES = "enumerate_sites"
    CALC_ADSORPTION = "calc_adsorption"
    CALC_DOS = "calc_dos"
    RUN_MD = "run_md"
    RUN_NEB = "run_neb"

class TaskNode(BaseModel):
    """Represents a single computational or analytical task in a workflow."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending" # pending, running, completed, failed
    result: Optional[Any] = None

    def __hash__(self):
        return hash(self.task_id)
        
    def __eq__(self, other):
        if not isinstance(other, TaskNode): return False
        return self.task_id == other.task_id

class WorkflowGraph:
    """
    A Directed Acyclic Graph (DAG) representing a scientific workflow.
    Ensures tasks are executed in the correct dependency order.
    """
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, TaskNode] = {}

    def add_task(self, task: TaskNode) -> str:
        """Adds a task to the DAG."""
        self.nodes[task.task_id] = task
        self.graph.add_node(task.task_id, task_type=task.task_type.value)
        return task.task_id

    def add_dependency(self, parent_id: str, child_id: str):
        """Specifies that child_id depends on the successful completion of parent_id."""
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError("Both tasks must be added to the graph before setting a dependency.")
        self.graph.add_edge(parent_id, child_id)
        
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent_id, child_id)
            raise ValueError("Adding this dependency creates a cycle, which is not allowed.")

    def get_execution_order(self) -> List[TaskNode]:
        """Returns tasks in topological order (safe for execution)."""
        try:
            ordered_ids = list(nx.topological_sort(self.graph))
            return [self.nodes[tid] for tid in ordered_ids]
        except nx.NetworkXUnfeasible:
            logger.error("Workflow graph contains cycles and cannot be executed.")
            return []

    def summarize(self) -> str:
        """Returns a string representation of the workflow DAG."""
        order = self.get_execution_order()
        if not order: return "Empty or invalid workflow."
        
        summary = f"Workflow: {self.name}\n"
        for i, task in enumerate(order):
            deps = list(self.graph.predecessors(task.task_id))
            dep_str = f" (Depends on: {', '.join(deps)})" if deps else ""
            summary += f"  {i+1}. [{task.task_type.value}] {task.task_id}{dep_str}\n"
        return summary

class WorkflowExecutor:
    """
    Engine to orchestrate the execution of a WorkflowGraph.
    Traverses the DAG and manages the passing of data between task nodes.
    """
    def __init__(self, strategist: Any):
        self.strategist = strategist

    def execute(self, graph: WorkflowGraph, sim_type: Any, iteration: int) -> Dict[str, Any]:
        """
        Executes the workflow graph in topological order.
        """
        logger.info(f"Executing Workflow DAG: {graph.name}")
        order = graph.get_execution_order()
        
        last_job_id = None
        
        for task in order:
            if task.status == "completed": continue
            logger.info(f"  -> [Task] {task.task_type.value} ({task.task_id})")
            
            # 1. Dispatch based on Type
            if task.task_type == TaskType.BUILD_SLAB:
                struct = self.strategist.builder.build_structure(self.strategist.pending_state)
                self.strategist.pending_state.slab_structure = struct
                task.status = "completed"
                task.result = struct
                
            elif task.task_type in [TaskType.RELAX_SLAB, TaskType.CALC_ADSORPTION, TaskType.RUN_MD]:
                state = self.strategist.pending_state
                job_id = self.strategist.compute.submit_job(state.slab_structure, state, sim_type=sim_type, iteration=iteration)
                
                # Check status
                from execution.compute_agent import JobStatus
                status = self.strategist.compute.get_job_status(job_id)
                
                if status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    task.status = "running"
                    task.result = job_id
                    # Return early to allow for async polling
                    return {"status": "pending", "job_id": job_id}
                
                task.status = "completed" if status == JobStatus.COMPLETED else "failed"
                task.result = job_id
                last_job_id = job_id

            elif task.task_type == TaskType.ENUMERATE_SITES:
                # Placeholder for site finder integration
                task.status = "completed"

        # 2. Collect Final Results
        if not last_job_id:
            raise ValueError("Workflow completed without a final job ID.")
            
        results_path = self.strategist.compute.fetch_results(last_job_id)
        eval_context = {"state": self.strategist.pending_state}
        observables, reward = self.strategist.evaluator.evaluate_calculation(results_path, eval_context)
        
        return {
            "state": self.strategist.pending_state,
            "reward": reward,
            "observables": observables,
            "status": "completed"
        }

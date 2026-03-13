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

    def get_ready_tasks(self) -> List[TaskNode]:
        """Returns tasks whose dependencies are all 'completed'."""
        ready = []
        for node_id in self.graph.nodes:
            task = self.nodes[node_id]
            if task.status != "pending":
                continue
            
            # Check if all parents are completed
            parents = list(self.graph.predecessors(node_id))
            all_parents_done = all(self.nodes[p].status == "completed" for p in parents)
            
            if all_parents_done:
                ready.append(task)
        return ready

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

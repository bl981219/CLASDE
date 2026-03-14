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
        next_state = self.strategist.current_state # Placeholder if needed
        
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

import networkx as nx
from typing import Dict, Any, List, Optional
from core.state import SurfaceState
from core.action import MutationAction

class ExperimentNode:
    """
    Represents a single scientific experiment within the Knowledge Graph.
    
    Unlike a basic MemoryGraph node (which just holds state and reward), an 
    ExperimentNode explicitly couples the physical `state` with the scientific 
    `objective` under which it was generated, the `action` taken to reach it, 
    and the full physical `result`.
    """
    def __init__(self, node_id: str, state: SurfaceState, action: Optional[MutationAction], 
                 objective: str, result: Dict[str, Any], metadata: Dict[str, Any] = None):
        self.node_id = node_id
        self.state = state
        self.action = action
        self.objective = objective
        self.result = result
        self.metadata = metadata or {}

class KnowledgeGraph:
    """
    A higher-level semantic graph for autonomous scientific reasoning.
    
    While the `MemoryGraph` tracks the immediate trajectory of the Bayesian Optimizer, 
    the `KnowledgeGraph` tracks experiments, hypotheses, and structural relationships 
    *across* multiple campaigns and objectives. 
    
    The Hypothesis Agent (PI) queries this graph to detect global structure-property 
    relationships (e.g., "Do oxygen vacancies always lower the d-band center across 
    different perovskite families?").
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.experiments: Dict[str, ExperimentNode] = {}
        
    def add_experiment(self, exp: ExperimentNode, parent_id: Optional[str] = None):
        self.experiments[exp.node_id] = exp
        self.graph.add_node(exp.node_id, data=exp)
        if parent_id and parent_id in self.experiments:
            self.graph.add_edge(parent_id, exp.node_id, relation="DERIVED_FROM")

    def get_experiments_by_objective(self, objective: str) -> List[ExperimentNode]:
        return [exp for exp in self.experiments.values() if exp.objective == objective]

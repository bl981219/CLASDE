import logging
from typing import Dict, Any, List, Optional
import json
import os
import networkx as nx
from science.experiment_graph import KnowledgeGraph, ScienceNode, NodeType, RelationType
from core.state import SurfaceState
from core.action import MutationAction

logger = logging.getLogger(__name__)

class KnowledgeGraphMemory:
    """
    Persistence layer for the scientific Knowledge Graph.
    
    Handles the serialization of nodes (Material, Surface, Structure, Calculation, Result) 
    and their scientific relationships.
    """
    def __init__(self, storage_path: str = "data/results/knowledge_graph.json") -> None:
        self.storage_path = storage_path
        
    def save(self, kg: KnowledgeGraph) -> None:
        """Serialize the KnowledgeGraph to disk."""
        data = {
            "nodes": [],
            "edges": []
        }
        
        # Serialize Nodes
        for node_id, node in kg.nodes.items():
            data["nodes"].append({
                "id": node_id,
                "type": node.node_type.value,
                "properties": node.properties
            })
            
        # Serialize Edges
        for u, v, attr in kg.graph.edges(data=True):
            data["edges"].append({
                "source": u,
                "target": v,
                "relation": attr.get("relation"),
                "metadata": {k: v for k, v in attr.items() if k != "relation"}
            })
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> KnowledgeGraph:
        """Load the KnowledgeGraph from disk."""
        kg = KnowledgeGraph()
        if not os.path.exists(self.storage_path):
            return kg
            
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load knowledge graph from {self.storage_path}: {e}")
            return kg
            
        # Load Nodes
        for n in data.get("nodes", []):
            node = ScienceNode(
                node_id=n["id"],
                node_type=NodeType(n["type"]),
                properties=n["properties"]
            )
            kg.add_node(node)
            
        # Load Edges
        for e in data.get("edges", []):
            kg.add_relation(
                source_id=e["source"],
                target_id=e["target"],
                relation=RelationType(e["relation"]),
                metadata=e.get("metadata")
            )
            
        return kg

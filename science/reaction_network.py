import networkx as nx
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ReactionNode:
    def __init__(self, species_id: str, is_transition_state: bool = False, energy: Optional[float] = None):
        self.species_id = species_id
        self.is_transition_state = is_transition_state
        self.energy = energy

class ReactionNetwork:
    """
    Manages the catalytic reaction pathways on a specific surface.
    Tracks intermediates (nodes) and elementary steps / barriers (edges).
    """
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        
    def add_species(self, species_id: str, energy: float) -> None:
        self.graph.add_node(species_id, obj=ReactionNode(species_id, energy=energy))
        
    def add_reaction_step(self, reactant_id: str, product_id: str, ts_id: Optional[str] = None, barrier: float = 0.0) -> None:
        if ts_id:
            self.graph.add_node(ts_id, obj=ReactionNode(ts_id, is_transition_state=True))
            self.graph.add_edge(reactant_id, ts_id, type="activation", barrier=barrier)
            self.graph.add_edge(ts_id, product_id, type="relaxation")
        else:
            self.graph.add_edge(reactant_id, product_id, type="elementary_step", barrier=barrier)
            
    def get_pathway_energy_profile(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Returns the lowest barrier pathway between two species."""
        try:
            path = nx.shortest_path(self.graph, start_id, end_id, weight="barrier")
            profile = []
            for node in path:
                node_data = self.graph.nodes[node]["obj"]
                profile.append({"species": node_data.species_id, "energy": node_data.energy, "is_ts": node_data.is_transition_state})
            return profile
        except nx.NetworkXNoPath:
            return []

import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ReactionIntermediate:
    """Represents a stable species on the surface or in gas phase."""
    def __init__(self, species_id: str, energy: float, stoichiometry: Dict[str, float], 
                 entropy: float = 0.0, enthalpy: Optional[float] = None):
        self.species_id = species_id
        self.energy = energy # DFT energy or similar
        self.stoichiometry = stoichiometry
        self.entropy = entropy
        self.enthalpy = enthalpy or energy

class ReactionStep:
    """Represents an elementary reaction step (e.g., A* + B* -> C* + D*)."""
    def __init__(self, reactant_ids: List[str], product_ids: List[str], 
                 barrier: float = 0.0, ts_energy: Optional[float] = None):
        self.reactant_ids = reactant_ids
        self.product_ids = product_ids
        self.barrier = barrier # Activation energy
        self.ts_energy = ts_energy

class ReactionNetwork:
    """
    Agent Domain Object: Reaction Network.
    
    Manages the catalytic reaction pathways on a specific surface.
    Tracks intermediates (nodes) and elementary steps / barriers (edges).
    This enables microkinetic modeling and catalytic cycle discovery.
    """
    def __init__(self, network_id: str) -> None:
        self.network_id = network_id
        self.graph = nx.DiGraph()
        self.intermediates: Dict[str, ReactionIntermediate] = {}
        self.steps: List[ReactionStep] = []

    def add_intermediate(self, intermediate: ReactionIntermediate) -> None:
        """Add a stable intermediate to the network."""
        self.intermediates[intermediate.species_id] = intermediate
        self.graph.add_node(intermediate.species_id, type="intermediate", properties={
            "energy": intermediate.energy,
            "stoichiometry": intermediate.stoichiometry
        })

    def add_step(self, step: ReactionStep) -> None:
        """Add an elementary step linking intermediates."""
        self.steps.append(step)
        # Simplified: link first reactant to first product for graph traversal
        # In real microkinetics, this is a bipartite graph or hypergraph
        source = "+".join(sorted(step.reactant_ids))
        target = "+".join(sorted(step.product_ids))
        
        # Ensure 'virtual' nodes for composite states exist
        if source not in self.graph:
            self.graph.add_node(source, type="state_composite")
        if target not in self.graph:
            self.graph.add_node(target, type="state_composite")
            
        self.graph.add_edge(source, target, type="elementary_step", barrier=step.barrier, ts_energy=step.ts_energy)

    def calculate_rate_constant(self, source: str, target: str, temperature: float) -> float:
        """Calculate Arrhenius rate constant: k = A * exp(-Ea / RT)."""
        edge_data = self.graph.get_edge_data(source, target)
        if not edge_data:
            return 0.0
            
        ea = edge_data["barrier"] # in eV
        kb = 8.617333262145e-5 # eV/K
        pre_exponential = 1e13 # standard frequency factor placeholder
        
        return float(pre_exponential * np.exp(-ea / (kb * temperature)))

    def get_reaction_profile(self, pathway: List[str]) -> List[Dict[str, Any]]:
        """Returns energies and barriers along a specified pathway."""
        profile = []
        for i in range(len(pathway) - 1):
            u, v = pathway[i], pathway[i+1]
            edge = self.graph.get_edge_data(u, v)
            u_data = self.graph.nodes[u]
            profile.append({
                "state": u,
                "energy": u_data.get("properties", {}).get("energy", 0.0),
                "barrier_to_next": edge.get("barrier", 0.0)
            })
        return profile

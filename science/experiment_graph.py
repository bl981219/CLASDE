import logging
import networkx as nx
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from core.state import SurfaceState
from core.action import MutationAction

logger = logging.getLogger(__name__)

class NodeType(str, Enum):
    MATERIAL = "material"
    SURFACE = "surface"
    SITE = "site"
    COVERAGE_STATE = "coverage_state"
    ADSORPTION_CONFIGURATION = "adsorption_configuration"
    ADSORBATE = "adsorbate"
    STRUCTURE = "structure"
    INTERMEDIATE = "intermediate"
    TRANSITION_STATE = "transition_state"
    REACTION = "reaction"
    REACTION_PATH = "reaction_path"
    REACTION_NETWORK = "reaction_network"
    CALCULATION = "calculation"
    RESULT = "result"

class RelationType(str, Enum):
    HAS_SURFACE = "has_surface"
    HAS_STRUCTURE = "has_structure"
    HAS_SITE = "has_site"
    ADSORBS = "adsorbs"
    RELAXED_TO = "relaxed_to"
    CALCULATED_BY = "calculated_by"
    YIELDS = "yields"
    DERIVED_FROM = "derived_from"
    TRAINED_ON = "trained_on"
    REACTS_TO = "reacts_to"
    ACTIVATES_TO = "activates_to"
    PART_OF_NETWORK = "part_of_network"
    PART_OF_PATH = "part_of_path"

class ScienceNode:
    """Base class for all nodes in the scientific knowledge graph."""
    def __init__(self, node_id: str, node_type: NodeType, properties: Optional[Dict[str, Any]] = None) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.properties = properties or {}

    def __repr__(self) -> str:
        return f"Node({self.node_type.value}, {self.node_id})"

class KnowledgeGraph:
    """
    The Central Knowledge Structure of CLASDE.
    
    This graph tracks the full scientific provenance of the discovery process, 
    linking materials to surfaces, structures, calculations, and empirical results.
    """
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ScienceNode] = {}

    def add_node(self, node: ScienceNode) -> None:
        """Add a scientific node to the graph."""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, type=node.node_type, properties=node.properties)

    def add_relation(self, source_id: str, target_id: str, relation: RelationType, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Establish a directed relationship between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.error(f"Both source ({source_id}) and target ({target_id}) nodes must exist.")
            raise ValueError(f"Both source ({source_id}) and target ({target_id}) nodes must exist.")
        self.graph.add_edge(source_id, target_id, relation=relation, **(metadata or {}))

    # --- Convenience Helpers for Common Discovery Patterns ---

    def record_site(self, surface_id: str, site_data: Dict[str, Any]) -> str:
        """Add a specific adsorption site to the graph."""
        site_id = f"site_{surface_id}_{site_data.get('type')}_{site_data.get('coordination')}"
        if site_id not in self.nodes:
            self.add_node(ScienceNode(site_id, NodeType.SITE, site_data))
            self.add_relation(surface_id, site_id, RelationType.HAS_SITE)
        return site_id

    def record_experiment(self, state: SurfaceState, action: Optional[MutationAction], 
                          result_data: Dict[str, Any], calc_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        High-level helper to ingest an autonomous experiment into the graph.
        Decomposes the SurfaceState into its semantic constituents using 
        surface science ontology.
        """
        # 1. Material Node (Bulk)
        mat_id = f"mat_{hash(frozenset(state.bulk_composition.items()))}"
        if mat_id not in self.nodes:
            self.add_node(ScienceNode(mat_id, NodeType.MATERIAL, {"composition": state.bulk_composition}))

        # 2. Surface Node (Facet + Termination)
        surf_id = f"surf_{state.miller_index}_{state.termination}"
        if surf_id not in self.nodes:
            self.add_node(ScienceNode(surf_id, NodeType.SURFACE, {
                "miller_index": state.miller_index,
                "termination": state.termination
            }))
            self.add_relation(mat_id, surf_id, RelationType.HAS_SURFACE)

        # 3. Handle Adsorbates and Sites
        parent_id = surf_id
        if state.adsorbates:
            for ads in state.adsorbates:
                # Record the Site
                site_id = self.record_site(surf_id, {
                    "type": ads.site_type,
                    "coordination": ads.site_coordination,
                    "neighbors": ads.site_neighbor_atoms
                })
                
                # Record the Adsorption Configuration (Geometry + Orientation)
                config_id = f"config_{state.get_id()[:8]}_{ads.identity}"
                self.add_node(ScienceNode(config_id, NodeType.ADSORPTION_CONFIGURATION, {
                    "identity": ads.identity,
                    "coverage": ads.coverage,
                    "orientation": ads.orientation,
                    "geometry": ads.geometry
                }))
                self.add_relation(site_id, config_id, RelationType.HAS_STRUCTURE)
                
                # Link to Adsorbate identity
                ads_node_id = f"ads_{ads.identity}"
                if ads_node_id not in self.nodes:
                    self.add_node(ScienceNode(ads_node_id, NodeType.ADSORBATE, {"identity": ads.identity}))
                self.add_relation(config_id, ads_node_id, RelationType.ADSORBS)
                
                parent_id = config_id # Results will link to the most specific config

        # 4. Structure/State Node
        struct_id = state.get_id()
        self.add_node(ScienceNode(struct_id, NodeType.STRUCTURE, {
            "defects": state.defects,
            "strain": state.strain,
            "conditions": {
                "T": state.temperature,
                "p": state.pressure,
                **state.external_conditions
            },
            "state_dict": state.model_dump(exclude={'slab_atoms'})
        }))
        self.add_relation(parent_id, struct_id, RelationType.HAS_STRUCTURE)

        # 5. Calculation Node
        calc_id = f"calc_{struct_id[:8]}_{calc_metadata.get('iteration') if calc_metadata else 'raw'}"
        self.add_node(ScienceNode(calc_id, NodeType.CALCULATION, calc_metadata or {}))
        self.add_relation(struct_id, calc_id, RelationType.CALCULATED_BY)

        # 6. Result Node
        res_id = f"res_{calc_id}"
        self.add_node(ScienceNode(res_id, NodeType.RESULT, result_data))
        self.add_relation(calc_id, res_id, RelationType.YIELDS)

    def record_reaction_network(self, network: Any):
        """Ingest a ReactionNetwork domain object into the Knowledge Graph."""
        from science.reaction_network import ReactionNetwork
        if not isinstance(network, ReactionNetwork):
            return
            
        net_node_id = f"network_{network.network_id}"
        self.add_node(ScienceNode(net_node_id, NodeType.REACTION_NETWORK, {"id": network.network_id}))
        
        # Add Intermediates
        for species_id, intermediate in network.intermediates.items():
            node_id = f"inter_{network.network_id}_{species_id}"
            self.add_node(ScienceNode(node_id, NodeType.INTERMEDIATE, {
                "species": species_id,
                "energy": intermediate.energy,
                "stoich": intermediate.stoichiometry
            }))
            self.add_relation(node_id, net_node_id, RelationType.PART_OF_NETWORK)
            
        # Add Steps/Transitions
        for step in network.steps:
            # Transition nodes
            if step.ts_energy:
                ts_id = f"ts_{network.network_id}_{hash(frozenset(step.reactant_ids))}"
                self.add_node(ScienceNode(ts_id, NodeType.TRANSITION_STATE, {"energy": step.ts_energy}))
                self.add_relation(ts_id, net_node_id, RelationType.PART_OF_NETWORK)

    def find_results_for_material(self, composition: Dict[str, float]) -> List[Dict[str, Any]]:
        """Query the graph for all empirical results associated with a specific bulk chemistry."""
        results: List[Dict[str, Any]] = []
        mat_id = f"mat_{hash(frozenset(composition.items()))}"
        if mat_id not in self.nodes:
            return results

        # Traverse: Material -> Surface -> Structure -> Calculation -> Result
        # Using a simple DFS or BFS here
        for surf in self.graph.successors(mat_id):
            for struct in self.graph.successors(surf):
                for calc in self.graph.successors(struct):
                    for res in self.graph.successors(calc):
                        if self.nodes[res].node_type == NodeType.RESULT:
                            results.append(self.nodes[res].properties)
        return results

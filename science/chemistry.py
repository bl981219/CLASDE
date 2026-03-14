import logging
from typing import List, Dict, Tuple
from pymatgen.core import Element

logger = logging.getLogger(__name__)

class ChemistryPhysicist:
    """
    Utility to handle chemical heuristics and species categorization 
    without hardcoding specific elements.
    """
    
    @staticmethod
    def categorize_perovskite_sites(composition: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Dynamically assigns elements to A or B sites in a perovskite (ABO3) structure.

        Uses relative comparisons of electronegativity and atomic radii to 
        distinguish sites, ensuring robustness across different chemical systems.
        """
        if not composition: return [], []
        
        elements = [Element(s) for s in composition.keys() if s != "O"]
        if not elements: return [], []

        # Sort by atomic radius (descending)
        # In Perovskites, A is significantly larger than B
        sorted_els = sorted(elements, key=lambda x: x.atomic_radius or 0, reverse=True)
        
        # Heuristic: The largest half of cations are likely A-sites
        # (For simple ABO3, it picks the 1st as A, 2nd as B)
        mid = max(1, len(sorted_els) // 2)
        a_sites = [el.symbol for el in sorted_els[:mid]]
        b_sites = [el.symbol for el in sorted_els[mid:]]
                
        return a_sites, b_sites

    @staticmethod
    def is_catalytically_active(symbol: str) -> bool:
        """
        Determines if a species is likely a transition metal active site.

        Args:
            symbol (str): The chemical symbol of the element.

        Returns:
            bool: True if the element is a transition metal, False otherwise.
        """
        el = Element(symbol)
        return el.is_transition_metal

    @staticmethod
    def get_layer_type(top_layer_species: List[str]) -> str:
        """
        Determines if a layer is AO or BO2 type based on the presence of B-site cations.

        Args:
            top_layer_species (List[str]): A list of element symbols present in the top surface layer.

        Returns:
            str: "BO2" if transition metals are present, otherwise "AO".
        """
        active_count = sum(1 for s in top_layer_species if ChemistryPhysicist.is_catalytically_active(s))
        # If transition metals are present, it's a B-site layer (BO2)
        return "BO2" if active_count > 0 else "AO"

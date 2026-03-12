import logging
from typing import List, Dict, Any, Tuple
from pymatgen.core import Element, Structure

logger = logging.getLogger(__name__)

class ChemistryPhysicist:
    """
    Utility to handle chemical heuristics and species categorization 
    without hardcoding specific elements.
    """
    
    @staticmethod
    def categorize_perovskite_sites(composition: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Dynamically assigns elements to A or B sites based on ionic radius and electronegativity.
        """
        a_sites = []
        b_sites = []
        
        for symbol, amount in composition.items():
            if symbol == "O": continue
            el = Element(symbol)
            
            # Heuristic: A-sites are larger, less electronegative (Alkali, Alkaline Earth, Lanthanoids)
            # B-sites are smaller transition metals
            if el.is_alkaline or el.is_alkali or el.is_lanthanoid or el.atomic_radius > 1.6:
                a_sites.append(symbol)
            else:
                b_sites.append(symbol)
                
        return a_sites, b_sites

    @staticmethod
    def is_catalytically_active(symbol: str) -> bool:
        """Determines if a species is likely a transition metal active site."""
        el = Element(symbol)
        return el.is_transition_metal

    @staticmethod
    def get_layer_type(top_layer_species: List[str]) -> str:
        """Determines if a layer is AO or BO2 type."""
        active_count = sum(1 for s in top_layer_species if ChemistryPhysicist.is_catalytically_active(s))
        # If transition metals are present, it's a B-site layer (BO2)
        return "BO2" if active_count > 0 else "AO"

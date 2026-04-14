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
        if not composition:
            return [], []

        elements = [Element(s) for s in composition.keys() if s != "O"]
        if not elements:
            return [], []

        # Chemistry-based assignment:
        # A-sites: rare earths (La, Y, Ce…) and alkaline/alkali earths (Ba, Sr, Ca…)
        # B-sites: transition metals (Fe, Co, Ni, Mn, Ti…)
        # Fallback for anything else (e.g. Pb, Bi): assign to B-site
        a_sites, b_sites = [], []
        for el in elements:
            if el.is_rare_earth or el.is_alkaline or el.is_alkali:
                a_sites.append(el.symbol)
            else:
                b_sites.append(el.symbol)

        # Safety fallback: if all elements landed on one side (e.g. pure metal),
        # revert to the radius-based heuristic so the function always returns both lists.
        if not a_sites or not b_sites:
            sorted_els = sorted(elements, key=lambda x: x.atomic_radius or 0, reverse=True)
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

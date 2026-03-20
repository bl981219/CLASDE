from typing import Any, Dict, List, Tuple
import logging
import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

logger = logging.getLogger(__name__)

class AdsorptionSiteFinder:
    """
    Identifies symmetry-unique adsorption sites on a given surface slab.
    """
    def find_sites(self, slab: Any) -> List[Dict[str, Any]]:
        """
        Analyzes the slab surface and returns available sites using Pymatgen.
        """
        try:
            logger.info("Scanning slab for adsorption sites using Pymatgen...")
            
            # 1. Standardize to Pymatgen Structure
            if hasattr(slab, "get_chemical_symbols"): # ASE Atoms
                struct = AseAtomsAdaptor.get_structure(slab)
            else:
                struct = slab
                
            asf = AdsorbateSiteFinder(struct)
            
            # 2. Find sites (default distance 2.0 A from surface)
            sites = asf.find_adsorption_sites(distance=2.0)
            
            found_sites = []
            # 3. Reduce and Categorize sites
            for site_type, coords_list in sites.items():
                for coords in coords_list:
                    # Map Pymatgen types to CLASDE internal types
                    mapped_type = site_type
                    if site_type == 'ontop': mapped_type = 'top'
                    
                    found_sites.append({
                        "type": mapped_type,
                        "position": tuple(coords),
                        "coordination": 1 if mapped_type == 'top' else (2 if mapped_type == 'bridge' else 3)
                    })
            
            if not found_sites:
                return self._geometric_fallback(slab)
                
            logger.info(f"Found {len(found_sites)} symmetry-unique sites.")
            return found_sites
            
        except Exception as e:
            logger.error(f"Pymatgen site finding failed: {e}. Using geometric fallback.")
            return self._geometric_fallback(slab)

    def _geometric_fallback(self, slab: Any) -> List[Dict[str, Any]]:
        """Simple geometric fallback if Pymatgen fails."""
        # Check if we have ASE or Pymatgen coordinates
        if hasattr(slab, "positions"):
            z_coords = slab.positions[:, 2]
            coords = slab.positions
        else:
            z_coords = slab.cart_coords[:, 2]
            coords = slab.cart_coords
            
        top_idx = np.where(z_coords > np.max(z_coords) - 0.5)[0]
        com = np.mean(coords[top_idx], axis=0)
        return [{"type": "top", "position": (com[0], com[1], com[2] + 1.8), "coordination": 1}]

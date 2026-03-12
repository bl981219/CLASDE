from typing import Any, Dict, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AdsorptionSiteFinder:
    """
    Identifies high-symmetry adsorption sites on a given surface slab.
    """
    def find_sites(self, slab: Any) -> List[Dict[str, Any]]:
        """
        Analyzes the slab surface and returns available sites using Pymatgen.
        """
        try:
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder
            from pymatgen.io.ase import AseAtomsAdaptor
            
            logger.info("Scanning slab for adsorption sites using Pymatgen...")
            
            # Convert ASE Atoms to Pymatgen Structure
            struct = AseAtomsAdaptor.get_structure(slab)
            asf = AdsorbateSiteFinder(struct)
            
            # Find sites (default distance 2.0 A from surface)
            sites = asf.find_adsorption_sites(distance=2.0)
            
            found_sites = []
            # Categorize sites
            for site_type, coords_list in sites.items():
                for coords in coords_list:
                    # In Pymatgen, site_type is 'ontop', 'bridge', 'hollow'
                    mapped_type = site_type
                    if site_type == 'ontop': mapped_type = 'top'
                    
                    found_sites.append({
                        "type": mapped_type,
                        "position": tuple(coords),
                        "coordination": 1 if mapped_type == 'top' else (2 if mapped_type == 'bridge' else 3)
                    })
            
            if not found_sites:
                logger.warning("No adsorption sites found by Pymatgen. Using geometric fallback.")
                return self._geometric_fallback(slab)
                
            return found_sites
            
        except Exception as e:
            logger.error(f"Pymatgen site finding failed: {e}. Using geometric fallback.")
            return self._geometric_fallback(slab)

    def _geometric_fallback(self, slab: Any) -> List[Dict[str, Any]]:
        """Simple geometric fallback if Pymatgen fails."""
        # Find the center of the top layer
        z_coords = slab.positions[:, 2]
        top_idx = np.where(z_coords > np.max(z_coords) - 0.5)[0]
        com = np.mean(slab.positions[top_idx], axis=0)
        return [{"type": "top", "position": (com[0], com[1], com[2] + 1.8), "coordination": 1}]

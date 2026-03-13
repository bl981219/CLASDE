import numpy as np
from typing import Any, Dict, List, Optional
import logging
from pymatgen.core import Structure, Element

logger = logging.getLogger(__name__)

class SurfaceDescriptors:
    """
    Utility class to compute physical and electronic descriptors for surface science.
    All methods are static to allow use across different agents.
    """
    
    @staticmethod
    def get_coordination_number(structure: Structure, site_idx: int, cutoff: float = 3.0) -> int:
        """Calculates effective coordination number using Pymatgen."""
        site = structure[site_idx]
        neighbors = structure.get_neighbors(site, r=cutoff)
        return len(neighbors)

    @staticmethod
    def get_generalized_coordination_number(structure: Structure, site_idx: int, cutoff: float = 3.0) -> float:
        """
        Calculates the Generalized Coordination Number (GCN).
        GCN = sum(cn_i / cn_max) for all neighbors i.
        """
        try:
            site = structure[site_idx]
            neighbors = structure.get_neighbors(site, r=cutoff)
            if not neighbors: return 0.0
            
            # Simple heuristic for cn_max: 12 for most close-packed systems
            # In future, this could be species-dependent from a bulk reference
            cn_max = 12.0
            
            gcn = 0.0
            for n in neighbors:
                n_cn = len(structure.get_neighbors(n, r=cutoff))
                gcn += n_cn / cn_max
            return float(gcn)
        except Exception as e:
            logger.warning(f"GCN calculation failed: {e}")
            return 0.0

    @staticmethod
    def calculate_band_moment(energies: np.ndarray, dos_vals: np.ndarray, order: int = 1) -> float:
        """
        Computes the n-th moment of a DOS distribution.
        Default order=1 is the band center.
        """
        # Only consider occupied states (below Fermi level at 0)
        mask = energies < 0
        total_dos = np.sum(dos_vals[mask])
        if total_dos == 0: return 0.0
        
        moment = np.sum((energies[mask]**order) * dos_vals[mask]) / total_dos
        return float(moment)

    @staticmethod
    def calculate_d_band_center(energies: np.ndarray, d_dos: np.ndarray) -> float:
        """Convenience wrapper for d-band center."""
        return SurfaceDescriptors.calculate_band_moment(energies, d_dos, order=1)

    @staticmethod
    def calculate_o2p_band_center(energies: np.ndarray, p_dos: np.ndarray) -> float:
        """Convenience wrapper for oxygen 2p-band center."""
        return SurfaceDescriptors.calculate_band_moment(energies, p_dos, order=1)

    @staticmethod
    def get_surface_atom_indices(structure: Structure, skin_depth: float = 1.5) -> List[int]:
        """Identifies indices of atoms in the surface layer."""
        z_coords = structure.cart_coords[:, 2]
        z_max = np.max(z_coords)
        return [i for i, z in enumerate(z_coords) if z > z_max - skin_depth]

import logging
import numpy as np
from typing import Dict, Tuple
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

logger = logging.getLogger(__name__)

class DomainValidator:
    """
    Scientific Validation Layer for CLASDE.
    Ensures thermodynamic and structural plausibility before execution.
    """

    @staticmethod
    def validate_slab(structure: Structure, 
                      min_layers: int = 4, 
                      min_vacuum: float = 15.0) -> Tuple[bool, str]:
        """
        Enforces physical constraints on surface slabs.

        Checks if the slab has sufficient vacuum to avoid spurious interactions
        between periodic images and ensures the slab is thick enough to represent
        bulk-like interior properties. Also warns about asymmetric slabs.

        Args:
            structure (Structure): The Pymatgen Structure object representing the slab.
            min_layers (int, optional): Minimum number of unique atomic layers required. Defaults to 4.
            min_vacuum (float, optional): Minimum vacuum size in Angstroms. Defaults to 15.0.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating validity and a message string.
        """
        # 1. Vacuum check
        lattice = structure.lattice
        # Assume vacuum is along c-axis
        z_coords = structure.cart_coords[:, 2]
        slab_thickness = np.max(z_coords) - np.min(z_coords)
        total_z = lattice.c
        vacuum_size = total_z - slab_thickness
        
        if vacuum_size < min_vacuum:
            return False, f"Insufficient vacuum: {vacuum_size:.2f} A < {min_vacuum} A"

        # 2. Symmetry check (Center of inversion for symmetric slabs)
        sga = SpacegroupAnalyzer(structure)
        if not sga.is_laue():
            # Warning only, as some physical models are intentionally asymmetric
            logger.warning("Slab lacks Laue symmetry (potential dipole moment).")

        # 3. Layer check
        # Heuristic: Count unique Z-planes
        z_planes = np.unique(np.round(z_coords, 2))
        if len(z_planes) < min_layers:
            return False, f"Insufficient layers: {len(z_planes)} < {min_layers}"

        return True, "Slab validated."

    @staticmethod
    def validate_charge_neutrality(composition: Dict[str, float]) -> Tuple[bool, str]:
        """
        Heuristic check for charge neutrality using common oxidation states.

        This prevents the system from exploring highly charged, unphysical states
        (e.g., massive oxygen deficiency without cation reduction).

        Args:
            composition (Dict[str, float]): A dictionary mapping element symbols to their stoichiometric amounts.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating charge neutrality (within a tolerance) and a message.
        """
        total_charge = 0.0
        for sym, amt in composition.items():
            el = Element(sym)
            # Use the most common oxidation state as a baseline
            try:
                ox_state = el.icsd_oxidation_states[0] if el.icsd_oxidation_states else 0
                total_charge += ox_state * amt
            except:
                continue
        
        if abs(total_charge) > 0.5:
            return False, f"Likely charge imbalance detected: Net charge ~{total_charge:.2f}"
        
        return True, "Charge balanced."

    @staticmethod
    def validate_adsorption_site(structure: Structure, site_idx: int) -> bool:
        """
        Ensures the adsorption site is on the surface and not buried in the bulk.

        Args:
            structure (Structure): The Pymatgen Structure object.
            site_idx (int): The index of the atom representing the adsorption site.

        Returns:
            bool: True if the site is on the surface (within 2.5 A of the highest Z-coordinate), False otherwise.
        """
        z_coords = structure.cart_coords[:, 2]
        z_max = np.max(z_coords)
        if structure[site_idx].coords[2] < z_max - 2.5:
            return False
        return True

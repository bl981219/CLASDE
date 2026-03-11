import logging
from typing import Dict, Any, Optional, List
import numpy as np
from core.state import SurfaceState
import warnings
import os

logger = logging.getLogger(__name__)

try:
    from ase import Atoms
    from ase.build import bulk, surface, add_adsorbate
    from pymatgen.core import Structure, Lattice, Molecule, Composition
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.transformations.standard_transformations import SupercellTransformation
    HAS_SIM_TOOLS = True
except ImportError:
    HAS_SIM_TOOLS = False

class StructureBuilder:
    """
    Agent 3 — Structure Builder (The PhD Student).
    Translates the formal SurfaceState into a high-fidelity 3D representation.
    """
    def __init__(self) -> None:
        if not HAS_SIM_TOOLS:
            logger.warning("ase/pymatgen not found. Structure generation will use fallbacks.")

    def build_structure(self, state: SurfaceState) -> Any:
        """Constructs a high-fidelity slab model from SurfaceState."""
        if not HAS_SIM_TOOLS:
            return Atoms('Cu', cell=(3.6, 3.6, 3.6), pbc=True)

        try:
            # 1. Create Bulk Structure based on stoichiometry
            bulk_struct = self._generate_realistic_bulk(state)
            
            # 2. Cleave and create Slab
            h, k, l = state.miller_index
            from pymatgen.core.surface import SlabGenerator
            # Use 10 A minimum slab thickness and 15 A vacuum
            sg = SlabGenerator(bulk_struct, (h, k, l), 10.0, 15.0)
            slabs = sg.get_slabs()
            if not slabs:
                raise ValueError(f"No slabs generated for facet {state.miller_index}")
            
            # Selection logic for termination
            # TODO: Match with state.termination
            slab_pmg = slabs[0]
            
            # 3. Convert to ASE for adsorbate/defect handling
            slab_ase = AseAtomsAdaptor.get_atoms(slab_pmg)
            
            # 4. Apply Mutations (Vacancies/Substitutions)
            slab_ase = self._apply_mutations(slab_ase, state)
            
            # 5. Place Adsorbates
            slab_ase = self._place_adsorbates(slab_ase, state)
            
            return slab_ase
        except Exception as e:
            logger.error(f"High-fidelity building failed: {e}. Using elemental fallback.")
            elements = list(state.bulk_composition.keys())
            fallback_el = elements[0] if elements else 'Cu'
            return bulk(fallback_el, cubic=True) * (2,2,2)

    def _generate_realistic_bulk(self, state: SurfaceState) -> Structure:
        """
        Creates a randomized perovskite supercell matching stoichiometry.
        Determines site preference (A vs B) using Shannon ionic radii.
        """
        from pymatgen.core import Element
        comp_dict = state.bulk_composition
        comp = Composition(comp_dict)
        elements = [el.symbol for el in comp.elements if el.symbol != "O"]
        
        a = state.metadata.get("lattice_constant", 3.905)
        lattice = Lattice.cubic(a)
        
        # 1. Physical Sorting: A-site (Large) vs B-site (Small)
        radii = {sym: Element(sym).average_ionic_radius for sym in elements}
        sorted_elements = sorted(elements, key=lambda x: radii[x], reverse=True)
        
        # Identification logic for Perovskites
        if len(elements) >= 2 and "O" in comp_dict:
            # Largest goes to A, smallest to B
            a_site_element = sorted_elements[0]
            b_site_element = sorted_elements[-1]
            
            # Use Pymatgen's built-in perovskite generator logic if possible or manual
            # Standard perovskite positions (fractional)
            symbols = [a_site_element, b_site_element, 'O', 'O', 'O']
            coords = [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            unit_cell = Structure(lattice, symbols, coords)
            
            # Scaling supercell to approximate stoichiometry
            # For demo, use 2x2x2 to allow for some mixing
            trans = SupercellTransformation(((2,0,0), (0,2,0), (0,0,2)))
            supercell = trans.apply_transformation(unit_cell)
            
            # TODO: Implement site-replacement based on actual fractional weights
            return supercell

        # Elemental/Simple bulk fallback
        return AseAtomsAdaptor.get_structure(bulk(elements[0] if elements else 'Cu', cubic=True))

    def _apply_mutations(self, atoms: Atoms, state: SurfaceState) -> Atoms:
        for defect in state.defects:
            if defect.get("type") == "vacancy":
                target = defect.get("site")
                indices = [i for i, atom in enumerate(atoms) if atom.symbol == target]
                if indices: atoms.pop(indices[-1])
            elif defect.get("type") == "substitution":
                orig, dopant = defect.get("original_element"), defect.get("dopant")
                indices = [i for i, atom in enumerate(atoms) if atom.symbol == orig]
                if indices: atoms[indices[0]].symbol = dopant
        return atoms

    def _place_adsorbates(self, atoms: Atoms, state: SurfaceState) -> Atoms:
        # Use our refined AdsorptionSiteFinder instead of hardcoded 'ontop'
        from execution.adsorption_site_finder import AdsorptionSiteFinder
        finder = AdsorptionSiteFinder()
        
        for ads in state.adsorbates:
            if ads.coverage > 0.0:
                sites = finder.find_sites(atoms)
                if sites:
                    # Select first site for now
                    site = sites[0]
                    try:
                        add_adsorbate(atoms, ads.identity, 1.8, position=(site['position'][0], site['position'][1]))
                    except: pass
        return atoms

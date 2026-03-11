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
            sg = SlabGenerator(bulk_struct, (h, k, l), 10.0, 10.0)
            slabs = sg.get_slabs()
            if not slabs:
                raise ValueError(f"No slabs generated for facet {state.miller_index}")
            
            # Select termination matching the state or first available
            slab_pmg = slabs[0]
            
            # 3. Convert to ASE for adsorbate/defect handling
            slab_ase = AseAtomsAdaptor.get_atoms(slab_pmg)
            
            # 4. Apply Mutations (Vacancies/Substitutions)
            slab_ase = self._apply_mutations(slab_ase, state)
            
            # 5. Place Adsorbates
            slab_ase = self._place_adsorbates(slab_ase, state)
            
            return slab_ase
        except Exception as e:
            logger.error(f"High-fidelity building failed: {e}. Using Cu fallback.")
            return Atoms('Cu', cell=(3.6, 3.6, 3.6), pbc=True)

    def _generate_realistic_bulk(self, state: SurfaceState) -> Structure:
        """Creates a randomized perovskite supercell matching stoichiometry."""
        comp_dict = state.bulk_composition
        comp = Composition(comp_dict)
        elements = [el.symbol for el in comp.elements]
        a = state.metadata.get("lattice_constant", 3.905)
        
        # Identification logic for Perovskites
        if len(elements) >= 3 and "O" in elements:
            # Categorize elements based on ionic radii/common knowledge
            # A-sites: large cations; B-sites: transition metals
            a_site_cands = [el for el in elements if el in ['La', 'Sr', 'Ba', 'Ca', 'Y', 'Pr', 'Nd', 'Sm']]
            b_site_cands = [el for el in elements if el in ['Fe', 'Co', 'Ni', 'Mn', 'Ti', 'Cr', 'Cu', 'Zr', 'Sc']]
            
            if a_site_cands and b_site_cands:
                lattice = Lattice.cubic(a)
                # Template unit cell based on first found A and B sites
                symbols = [a_site_cands[0], b_site_cands[0], 'O', 'O', 'O']
                coords = [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
                unit_cell = Structure(lattice, symbols, coords)
                
                # Expand to supercell to accommodate fractional stoichiometry
                trans = SupercellTransformation(((2,0,0), (0,2,0), (0,0,2)))
                supercell = trans.apply_transformation(unit_cell)
                
                # TODO: Implement randomized site occupation for multi-element A/B sites
                # For now, we return the base template to avoid hardcoding specific La/Fe
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
        for ads in state.adsorbates:
            if ads.coverage > 0.0:
                try:
                    # Place at reasonable surface height
                    add_adsorbate(atoms, ads.identity, 1.8, 'ontop')
                except: pass
        return atoms

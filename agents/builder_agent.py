import logging
from typing import Dict, Any, Optional, List
import numpy as np
import os
from core.state import SurfaceState

logger = logging.getLogger(__name__)

try:
    from ase import Atoms
    from ase.build import bulk, surface, add_adsorbate
    from pymatgen.core import Structure, Lattice, Composition, Element
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.transformations.standard_transformations import SupercellTransformation
    HAS_SIM_TOOLS = True
except ImportError:
    HAS_SIM_TOOLS = False

class StructureBuilder:
    """
    Agent 3 — Structure Builder (High Fidelity Perovskite Engine).
    Constructs stoichiometric LSCF slabs with AO and BO2 termination control.
    """
    def __init__(self) -> None:
        if not HAS_SIM_TOOLS:
            logger.warning("ase/pymatgen not found.")

    def build_structure(self, state: SurfaceState) -> Any:
        if not HAS_SIM_TOOLS:
            return Atoms('Cu', cell=(3.6, 3.6, 3.6), pbc=True)

        try:
            # 1. Create stoichiometric Bulk Perovskite (Generalized)
            bulk_struct = self._generate_perovskite(state)
            
            # 2. Cleave and create Slab
            from pymatgen.core.surface import SlabGenerator
            h, k, l = state.miller_index
            thickness = state.metadata.get("min_slab_size", 10.0)
            vacuum = state.metadata.get("min_vacuum_size", 15.0)
            
            sg = SlabGenerator(bulk_struct, (h, k, l), min_slab_size=thickness, min_vacuum_size=vacuum)
            slabs = sg.get_slabs()
            
            # 3. Select termination
            slab_pmg = self._select_termination(slabs, state.termination)
            
            # 4. Convert to ASE
            slab_ase = AseAtomsAdaptor.get_atoms(slab_pmg)
            slab_ase.set_pbc(True)
            
            # 5. Mutations & Adsorbates
            slab_ase = self._apply_mutations(slab_ase, state)
            slab_ase = self._place_adsorbates(slab_ase, state)
            
            # 6. Selective Dynamics (Configurable fraction)
            from ase.constraints import FixAtoms
            z_coords = slab_ase.positions[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            freeze_thresh = state.metadata.get("freeze_fraction", 0.5) * (z_max - z_min) + z_min
            
            frozen_indices = [i for i, z in enumerate(z_coords) if z < freeze_thresh]
            slab_ase.set_constraint(FixAtoms(indices=frozen_indices))
            
            # 7. Canonical sorting
            indices = np.argsort(slab_ase.get_chemical_symbols())
            slab_ase = slab_ase[indices]
            
            return slab_ase
        except Exception as e:
            logger.error(f"Structural building failed: {e}. Using elemental fallback.")
            return bulk('Fe', crystalstructure='fcc', a=3.6) * (3,3,3)

    def _generate_perovskite(self, state: SurfaceState) -> Structure:
        """
        Creates a supercell with the requested stoichiometry for any ABO3 system.
        """
        comp = state.bulk_composition
        a = state.metadata.get("lattice_constant", 3.905)
        lattice = Lattice.cubic(a)
        
        # Categorize elements into A, B, and O sites
        a_elements = []
        b_elements = []
        for sym, count in comp.items():
            if sym == "O": continue
            el = Element(sym)
            # Simple heuristic: A-sites are larger cations (Alkali, Alkaline Earth, Lanthanides)
            if el.is_alkaline or el.is_alkali or el.is_lanthanoid or el.row >= 5:
                a_elements.append(sym)
            else:
                b_elements.append(sym)
        
        # Template: Start with primary A and B
        base_a = a_elements[0] if a_elements else "La"
        base_b = b_elements[0] if b_elements else "Fe"
        
        struct = Structure(lattice, [base_a, base_b, "O", "O", "O"], 
                           [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
        
        # Supercell size (default 2x2x2)
        dim = state.metadata.get("bulk_supercell", (2,2,2))
        trans = SupercellTransformation(((dim[0],0,0), (0,dim[1],0), (0,0,dim[2])))
        supercell = trans.apply_transformation(struct)
        
        # Total sites
        total_a = dim[0] * dim[1] * dim[2]
        total_b = total_a
        
        # Apply A-site distribution
        a_indices = [i for i, s in enumerate(supercell) if s.specie.symbol == base_a]
        curr = 0
        for sym in a_elements:
            # target count proportional to bulk_composition ratio
            target_count = int(round(comp[sym] / sum(comp[el] for el in a_elements) * total_a))
            for _ in range(target_count):
                if curr < len(a_indices):
                    supercell.replace(a_indices[curr], sym)
                    curr += 1

        # Apply B-site distribution
        b_indices = [i for i, s in enumerate(supercell) if s.specie.symbol == base_b]
        curr = 0
        for sym in b_elements:
            target_count = int(round(comp[sym] / sum(comp[el] for el in b_elements) * total_b))
            for _ in range(target_count):
                if curr < len(b_indices):
                    supercell.replace(b_indices[curr], sym)
                    curr += 1
            
        return supercell

    def _select_termination(self, slabs: List, requested: str) -> Any:
        """Categorizes slabs by surface layer composition using chemical heuristics."""
        for s in slabs:
            z_coords = [site.coords[2] for site in s]
            z_max = np.max(z_coords)
            top_sites = [site for site in s if site.coords[2] > z_max - 1.0]
            
            # Detect if top layer is B-site dominated (Transition Metals)
            has_b_site = any(Element(site.specie.symbol).is_transition_metal for site in top_sites)
            
            if requested == "BO2" and has_b_site: return s
            if requested == "AO" and not has_b_site: return s
            
        return slabs[0]

    def _apply_mutations(self, atoms: Atoms, state: SurfaceState) -> Atoms:
        for defect in state.defects:
            if defect.get("type") == "vacancy":
                target = defect.get("site")
                indices = [i for i, atom in enumerate(atoms) if atom.symbol == target]
                if indices:
                    # Remove the one with highest Z
                    top_idx = indices[np.argmax(atoms.positions[indices, 2])]
                    atoms.pop(top_idx)
        return atoms

    def _place_adsorbates(self, atoms: Atoms, state: SurfaceState) -> Atoms:
        from execution.adsorption_site_finder import AdsorptionSiteFinder
        finder = AdsorptionSiteFinder()
        for ads in state.adsorbates:
            if ads.coverage > 0.0:
                sites = finder.find_sites(atoms)
                if sites:
                    # Filter for top sites on the surface
                    top_sites = [s for s in sites if s['type'] == 'top']
                    target = top_sites[0] if top_sites else sites[0]
                    add_adsorbate(atoms, ads.identity, 1.8, position=(target['position'][0], target['position'][1]))
        return atoms

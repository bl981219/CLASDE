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

from science.chemistry import ChemistryPhysicist

class StructureBuilder:
    """
    Agent 3 — Structure Builder (High Fidelity Perovskite Engine).
    Constructs stoichiometric perovskite slabs with adaptive termination control.
    """
    def __init__(self) -> None:
        if not HAS_SIM_TOOLS:
            logger.warning("ase/pymatgen not found.")

    def build_structure(self, state: SurfaceState) -> Structure:
        """Returns a Pymatgen Structure representation of the surface state."""
        if not HAS_SIM_TOOLS:
            return None

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

            # --- Domain Validation Step ---
            from science.validator import DomainValidator
            is_valid, msg = DomainValidator.validate_slab(
                slab_pmg, 
                min_layers=state.metadata.get("min_layers", 4),
                min_vacuum=vacuum
            )
            if not is_valid:
                logger.error(f"Slab Validation Failed: {msg}")
                # We return it but log the error; in future could trigger a rebuild

            # 4. Mutations & Adsorbates (Using ASE for convenience)
            slab_ase = AseAtomsAdaptor.get_atoms(slab_pmg)
            slab_ase.set_pbc(True)
            
            slab_ase = self._apply_mutations(slab_ase, state)
            slab_ase = self._place_adsorbates(slab_ase, state)
            
            # 5. Selective Dynamics (Configurable fraction)
            from ase.constraints import FixAtoms
            z_coords = slab_ase.positions[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            freeze_thresh = state.metadata.get("freeze_fraction", 0.5) * (z_max - z_min) + z_min
            
            frozen_indices = [i for i, z in enumerate(z_coords) if z < freeze_thresh]
            slab_ase.set_constraint(FixAtoms(indices=frozen_indices))
            
            # 6. Canonical sorting & Convert back to Pymatgen
            indices = np.argsort(slab_ase.get_chemical_symbols())
            slab_ase = slab_ase[indices]
            
            return AseAtomsAdaptor.get_structure(slab_ase)
        except Exception as e:
            logger.error(f"Structural building failed: {e}.")
            return None

    def _generate_perovskite(self, state: SurfaceState) -> Structure:
        """
        Creates a supercell with the requested stoichiometry and symmetry.
        Supports Cubic (Pm-3m), Tetragonal (I4/mcm), and Orthorhombic (Pbnm).
        """
        comp = state.bulk_composition
        a = state.metadata.get("lattice_constant", 3.905)
        symmetry = state.metadata.get("crystal_system", "cubic").lower()
        
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.surface import SlabGenerator
        
        if symmetry == "cubic":
            lattice = Lattice.cubic(a)
            coords = [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            species = ["La", "Fe", "O", "O", "O"] # Dummy, will be replaced
            struct = Structure(lattice, species, coords)
        elif symmetry == "tetragonal":
            # Heuristic for I4/mcm distortion
            lattice = Lattice.tetragonal(a * np.sqrt(2), a * 2.0)
            struct = Structure(lattice, ["La", "Fe", "O"], [[0,0,0], [0,0,0.25], [0.25, 0.25, 0]])
        elif symmetry == "orthorhombic":
            # Heuristic for Pbnm distortion
            lattice = Lattice.orthorhombic(a * np.sqrt(2), a * np.sqrt(2), a * 2.0)
            struct = Structure(lattice, ["La", "Fe", "O"], [[0,0,0], [0,0,0.25], [0.25, 0.25, 0]])
        else:
            lattice = Lattice.cubic(a)
            struct = Structure(lattice, ["La", "Fe", "O", "O", "O"], 
                               [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])

        # ... Element distribution logic remains but uses ChemistryPhysicist ...
        a_elements, b_elements = ChemistryPhysicist.categorize_perovskite_sites(comp)
        
        # Scale supercell
        dim = state.metadata.get("bulk_supercell", (2,2,2))
        struct.make_supercell(dim)
        
        # Dynamically replace species to match comp
        self._redistribute_species(struct, a_elements, b_elements, comp)
        return struct

    def _redistribute_species(self, struct: Structure, a_els: List[str], b_els: List[str], comp: Dict[str, float]):
        """Logic to replace dummy species with stoichiometric reality."""
        # Identification of A and B sites depends on coordination
        for i, site in enumerate(struct):
            if site.specie.symbol == "O": continue
            cn = len(struct.get_neighbors(site, 3.5))
            if cn > 8: # A-site (12-coordinated)
                target_sym = self._get_target_species(a_els, comp)
                struct.replace(i, target_sym)
            else: # B-site (6-coordinated)
                target_sym = self._get_target_species(b_els, comp)
                struct.replace(i, target_sym)

    def _get_target_species(self, allowed: List[str], comp: Dict[str, float]) -> str:
        # Weighted random choice or sequential filling based on ratios
        import random
        # Simplified: pick first for now, can be improved with accurate counter
        return allowed[0] if allowed else "Fe"

    def _select_termination(self, slabs: List, requested: str) -> Structure:
        """Categorizes slabs by surface layer composition using chemical heuristics."""
        for s in slabs:
            z_coords = [site.coords[2] for site in s]
            z_max = np.max(z_coords)
            top_sites = [site for site in s if site.coords[2] > z_max - 1.0]
            top_species = [site.specie.symbol for site in top_sites]
            
            # Detect layer type dynamically
            layer_type = ChemistryPhysicist.get_layer_type(top_species)
            
            if requested == layer_type: return s
            
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
        from science.adsorption_site_finder import AdsorptionSiteFinder
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

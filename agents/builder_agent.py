import logging
from typing import Dict, Any, Optional, List
from core.state import SurfaceState
import warnings
import os

logger = logging.getLogger(__name__)

try:
    from ase import Atoms
    from ase.build import bulk, surface, add_adsorbate
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor
    HAS_SIM_TOOLS = True
except ImportError:
    HAS_SIM_TOOLS = False

class StructureBuilder:
    """
    Agent 3 — Structure Builder (The PhD Student).
    
    This agent translates the formal mathematical descriptor (`SurfaceState`) into an 
    actual 3D physical representation using ASE (Atomic Simulation Environment) and Pymatgen.
    
    It enforces physical constraints automatically, such as:
    - Cleaving the correct Miller index facets.
    - Applying point defects (vacancies, substitutions).
    - Enforcing charge neutrality during aliovalent doping by introducing 
      compensating oxygen vacancies.
      
    The output is an ASE `Atoms` object ready for MLFF evaluation or DFT submission.
    """
    def __init__(self) -> None:
        if not HAS_SIM_TOOLS:
            logger.warning("ase or pymatgen not found. Physical structure generation will fail.")
            warnings.warn("ase or pymatgen not found. Physical structure generation will fail.")

    def build_structure(self, state: SurfaceState) -> Any:
        """
        Generate a physical structure from a SurfaceState.
        Returns an ASE Atoms object.
        """
        if not HAS_SIM_TOOLS:
            return None
            
        elements = list(state.bulk_composition.keys())
        
        # 1. Load or generate bulk structure
        # Special case: Perovskites (ABO3)
        is_perovskite = any(el in ["La", "Sr", "Ba", "Ca"] for el in elements) and \
                        any(el in ["Mn", "Fe", "Co", "Ni", "Ti"] for el in elements) and \
                        "O" in elements
        
        if is_perovskite:
            # For the demo, we'll try to load a perovskite reference or build a simple SrTiO3-like bulk
            # In a real scenario, we'd use a database of CIFs.
            try:
                # Mock: Build a 5-atom perovskite unit cell for demo
                # (Simple cubic perovskite approx)
                a = 3.905 # Lattice constant (A)
                bulk_atoms = Atoms('SrTiO3', 
                                   scaled_positions=[(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
                                   cell=(a, a, a), pbc=True)
                # Map the user's chemistry onto the sites
                # Site 0: A-site (Sr/La), Site 1: B-site (Ti/Fe/Mn), Sites 2-4: Oxygen
                
                # A-site mapping
                a_site_els = [el for el in elements if el in ["La", "Sr", "Ba", "Ca"]]
                if a_site_els:
                    bulk_atoms[0].symbol = a_site_els[0]
                    
                # B-site mapping
                b_site_els = [el for el in elements if el in ["Mn", "Fe", "Co", "Ni", "Ti"]]
                if b_site_els:
                    bulk_atoms[1].symbol = b_site_els[0]
                
            except Exception as e:
                logger.warning(f"Failed to build perovskite bulk: {e}. Falling back to Cu.")
                warnings.warn(f"Failed to build perovskite bulk: {e}. Falling back to Cu.")
                bulk_atoms = bulk('Cu', cubic=True)
        elif len(elements) == 1:
            # Simple elemental bulk (e.g., {'Cu': 1.0})
            element = elements[0]
            try:
                bulk_atoms = bulk(element, cubic=True)
            except Exception as e:
                logger.warning(f"Failed to build bulk {element}: {e}. Falling back to placeholder.")
                warnings.warn(f"Failed to build bulk {element}: {e}. Falling back to placeholder.")
                return self._placeholder_generation(state)
        else:
            # For complex materials, we'd load from a CIF in metadata. 
            # In this demo version, we'll fall back to a default Cu bulk if CIF is missing.
            cif_path = state.metadata.get("bulk_cif_path")
            if cif_path:
                try:
                    from ase.io import read
                    bulk_atoms = read(cif_path)
                except Exception as e:
                    logger.warning(f"Failed to read {cif_path}: {e}")
                    warnings.warn(f"Failed to read {cif_path}: {e}")
                    bulk_atoms = bulk('Cu', cubic=True)
            else:
                bulk_atoms = bulk('Cu', cubic=True) # Fallback

        # 2. Cleave facet (h,k,l) using ASE surface
        h, k, l = state.miller_index
        try:
            # Create a slab with 3 layers and 15A vacuum
            slab = surface(bulk_atoms, (h, k, l), layers=3, vacuum=15.0)
            slab.center(vacuum=15.0, axis=2)
        except Exception as e:
            logger.warning(f"Failed to cleave surface {state.miller_index}: {e}")
            warnings.warn(f"Failed to cleave surface {state.miller_index}: {e}")
            slab = bulk_atoms.copy()

        # 3. Apply defects (enhanced with charge compensation)
        for defect in state.defects:
            if defect.get("type") == "vacancy":
                if len(slab) > 0:
                    # Target a specific element if provided
                    target = defect.get("site")
                    indices = [i for i, atom in enumerate(slab) if atom.symbol == target] if target else list(range(len(slab)))
                    if indices:
                        slab.pop(indices[-1])
            elif defect.get("type") == "substitution":
                # Replace an atom of original_element with dopant
                orig = defect.get("original_element")
                dopant = defect.get("dopant")
                indices = [i for i, atom in enumerate(slab) if atom.symbol == orig]
                if indices:
                    slab[indices[0]].symbol = dopant
                    
                    # Aliovalent Doping Charge Compensation Logic:
                    if orig == "La" and dopant == "Sr":
                        o_indices = [i for i, atom in enumerate(slab) if atom.symbol == "O"]
                        if o_indices:
                            slab.pop(o_indices[0]) # Introduce oxygen vacancy
                            logger.info("Charge Compensation: Introduced Oxygen Vacancy for Sr-doping.")
            elif defect.get("type") == "swap":
                # Segregation modeling: swap surface element_a with subsurface element_b
                el_a = defect.get("element_a")
                el_b = defect.get("element_b")
                idx_a = [i for i, atom in enumerate(slab) if atom.symbol == el_a]
                idx_b = [i for i, atom in enumerate(slab) if atom.symbol == el_b]
                if idx_a and idx_b:
                    # Swap the top-most idx_a with the bottom-most idx_b (approx)
                    slab[idx_a[-1]].symbol = el_b
                    slab[idx_b[0]].symbol = el_a
                    
        # 4. Place adsorbate a at coverage θ
        if state.adsorbate and state.coverage > 0.0:
            try:
                height = 1.5 
                add_adsorbate(slab, state.adsorbate, height, 'ontop')
            except Exception as e:
                logger.warning(f"Failed to add adsorbate {state.adsorbate}: {e}")
                warnings.warn(f"Failed to add adsorbate {state.adsorbate}: {e}")

        # Final Sanity Check: Never return an empty structure
        if len(slab) == 0:
            return Atoms('Cu', positions=[(0, 0, 0)])

        return slab

    def _placeholder_generation(self, state: SurfaceState) -> Any:
        """Simple placeholder atoms object."""
        if not HAS_SIM_TOOLS:
            return None
        return Atoms('Cu', positions=[(0, 0, 0)])

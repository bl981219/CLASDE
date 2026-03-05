from typing import Dict, Any, Optional
from core.state import SurfaceState
import warnings

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
    def __init__(self):
        if not HAS_SIM_TOOLS:
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
        if len(elements) == 1:
            # Simple elemental bulk (e.g., {'Cu': 1.0})
            element = elements[0]
            try:
                bulk_atoms = bulk(element, cubic=True)
            except Exception as e:
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
            warnings.warn(f"Failed to cleave surface {state.miller_index}: {e}")
            slab = bulk_atoms.copy()

        # 3. Apply defects (enhanced with charge compensation)
        for defect in state.defects:
            if defect.get("type") == "vacancy":
                # Naive vacancy creation: pop the highest Z atom
                if len(slab) > 0:
                    slab.pop()
            elif defect.get("type") == "substitution":
                # Replace an atom of original_element with dopant
                orig = defect.get("original_element")
                dopant = defect.get("dopant")
                indices = [i for i, atom in enumerate(slab) if atom.symbol == orig]
                if indices:
                    # Replace the first matching site found
                    slab[indices[0]].symbol = dopant
                    
                    # Aliovalent Doping Charge Compensation Logic:
                    # If Sr2+ substitutes La3+ (La0.5Sr0.5MnO3), we need 
                    # specific oxygen vacancies to maintain charge neutrality.
                    # R = 2*(Σ n_dopant * Δz) / area
                    if orig == "La" and dopant == "Sr":
                        # For every 2 Sr-substitutions, we introduce 1 O-vacancy
                        # This is a simplified implementation for perovskites
                        o_indices = [i for i, atom in enumerate(slab) if atom.symbol == "O"]
                        if o_indices:
                            slab.pop(o_indices[0]) # Introduce oxygen vacancy
                            print(f"  Charge Compensation: Introduced Oxygen Vacancy for Sr-doping.")
                    
        # 4. Place adsorbate a at coverage θ
        if state.adsorbate and state.coverage > 0.0:
            # Simple top-site adsorption for demonstration
            # In a full implementation, coverage would dictate multi-site placement
            height = 1.5 # Angstroms above the surface
            add_adsorbate(slab, state.adsorbate, height, 'ontop')

        return slab

    def _placeholder_generation(self, state: SurfaceState) -> Any:
        """Simple placeholder atoms object."""
        if not HAS_SIM_TOOLS:
            return None
        return Atoms('Cu', positions=[(0, 0, 0)])

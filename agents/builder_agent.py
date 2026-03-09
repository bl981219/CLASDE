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
    HAS_SIM_TOOLS = True
except ImportError:
    HAS_SIM_TOOLS = False

class StructureBuilder:
    """
    Agent 3 — Structure Builder (The PhD Student).
    
    This agent translates the formal mathematical descriptor (`SurfaceState`) into an 
    actual 3D physical representation using ASE (Atomic Simulation Environment) and Pymatgen.
    
    V2: Generalized architecture supporting CIF loading and robust procedural generation.
    """
    def __init__(self) -> None:
        if not HAS_SIM_TOOLS:
            logger.warning("ase not found. Physical structure generation will fail.")

    def build_structure(self, state: SurfaceState) -> Any:
        """
        Generate a physical structure from a SurfaceState.
        Returns an ASE Atoms object.
        """
        if not HAS_SIM_TOOLS:
            return None
            
        # 1. Load or generate bulk structure
        bulk_atoms = None
        
        # Priority A: Load from metadata CIF if provided
        cif_path = state.metadata.get("bulk_cif_path")
        if cif_path and os.path.exists(cif_path):
            try:
                from ase.io import read
                bulk_atoms = read(cif_path)
                logger.info(f"Loaded bulk structure from {cif_path}")
            except Exception as e:
                logger.warning(f"Failed to read CIF at {cif_path}: {e}")

        # Priority B: Procedural Generation based on stoichiometry
        if bulk_atoms is None:
            elements = sorted(list(state.bulk_composition.keys()))
            
            # Sub-Priority B1: Standard Perovskite Detection (ABO3)
            # General check: 3+ elements, contains Oxygen
            is_perovskite = len(elements) >= 3 and "O" in elements
            if is_perovskite:
                try:
                    a = state.metadata.get("lattice_constant", 3.905)
                    # We create a generic ABO3 template
                    # Map sorted elements: 0 -> A, 1 -> B, 2 -> O
                    bulk_atoms = Atoms(symbols=[elements[0], elements[1], 'O', 'O', 'O'], 
                                       scaled_positions=[(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
                                       cell=(a, a, a), pbc=True)
                except Exception as e:
                    logger.debug(f"Perovskite template failed: {e}")

            # Sub-Priority B2: Simple Elemental Bulk
            if bulk_atoms is None and len(elements) == 1:
                try:
                    bulk_atoms = bulk(elements[0], cubic=True)
                except: pass

        # Priority C: Absolute Fallback
        if bulk_atoms is None:
            logger.warning("All bulk generation methods failed. Using Cu fallback.")
            bulk_atoms = bulk('Cu', cubic=True)

        # 2. Cleave facet (h,k,l)
        h, k, l = state.miller_index
        if h == 0 and k == 0 and l == 0: l = 1 
        
        try:
            # We use a 3-layer slab with 15A vacuum
            slab = surface(bulk_atoms, (h, k, l), layers=3, vacuum=15.0)
            slab.center(vacuum=15.0, axis=2)
        except Exception as e:
            logger.warning(f"Surface cleave failed: {e}")
            slab = bulk_atoms.repeat((2,2,2))
            slab.center(vacuum=15.0, axis=2)

        # Final sanity check on slab integrity
        if len(slab) == 0 or slab.cell is None or np.any(np.linalg.norm(slab.cell, axis=1) == 0):
            slab = Atoms('Cu', cell=(3.6, 3.6, 3.6), positions=[(0,0,0)], pbc=True)

        # 3. Apply defects (Dynamic mapping)
        for defect in state.defects:
            try:
                if defect.get("type") == "vacancy" and len(slab) > 0:
                    target = defect.get("site")
                    indices = [i for i, atom in enumerate(slab) if atom.symbol == target] if target else list(range(len(slab)))
                    if indices: slab.pop(indices[-1])
                elif defect.get("type") == "substitution" and len(slab) > 0:
                    orig, dopant = defect.get("original_element"), defect.get("dopant")
                    indices = [i for i, atom in enumerate(slab) if atom.symbol == orig]
                    if indices: slab[indices[0]].symbol = dopant
            except Exception as e:
                logger.debug(f"Defect application failed: {e}")
                    
        # 4. Place adsorbates
        for ads in state.adsorbates:
            if ads.coverage > 0.0 and len(slab) > 0:
                try:
                    height = 1.5 
                    # Try named site first, if it fails use absolute coordinates
                    try:
                        add_adsorbate(slab, ads.identity, height, ads.site_type if ads.site_type == 'ontop' else 'ontop')
                    except:
                        com = np.mean(slab.positions, axis=0)
                        add_adsorbate(slab, ads.identity, height, position=(com[0], com[1]))
                except Exception as e:
                    logger.warning(f"Adsorbate placement failed: {e}")

        return slab

    def _placeholder_generation(self, state: SurfaceState) -> Any:
        return Atoms('Cu', cell=(3.6, 3.6, 3.6), pbc=True)

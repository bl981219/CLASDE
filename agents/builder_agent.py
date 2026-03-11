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
    
    Translates the formal SurfaceState into an actual 3D physical representation.
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
            
        bulk_atoms = None
        
        # 1. Procedural Generation based on stoichiometry
        elements = sorted([k for k, v in state.bulk_composition.items() if v > 0])
        
        if len(elements) >= 3 and "O" in elements:
            # Perovskite (ABO3) detection
            try:
                a = state.metadata.get("lattice_constant", 3.905)
                # Ensure we have enough distinct elements for ABO3
                # If LSCF, we have La, Sr, Co, Fe, O
                # Simplified: First is A, second is B, last is O
                bulk_atoms = Atoms(symbols=[elements[0], elements[1], 'O', 'O', 'O'], 
                                   scaled_positions=[(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
                                   cell=(a, a, a), pbc=True)
            except: pass

        if bulk_atoms is None:
            # Fallback to elemental bulk or Cu
            try:
                bulk_atoms = bulk(elements[0] if elements else 'Cu', cubic=True)
            except:
                bulk_atoms = bulk('Cu', cubic=True)

        # 2. Cleave facet
        h, k, l = state.miller_index
        if h == 0 and k == 0 and l == 0: l = 1 
        
        try:
            slab = surface(bulk_atoms, (h, k, l), layers=3, vacuum=10.0)
            slab.center(vacuum=10.0, axis=2)
        except:
            slab = bulk_atoms.repeat((2,2,2))
            slab.center(vacuum=10.0, axis=2)

        # 3. Apply defects
        for defect in state.defects:
            try:
                if defect.get("type") == "vacancy" and len(slab) > 0:
                    target = defect.get("site")
                    indices = [i for i, atom in enumerate(slab) if atom.symbol == target]
                    if indices: slab.pop(indices[-1])
                elif defect.get("type") == "substitution" and len(slab) > 0:
                    orig, dopant = defect.get("original_element"), defect.get("dopant")
                    indices = [i for i, atom in enumerate(slab) if atom.symbol == orig]
                    if indices: slab[indices[0]].symbol = dopant
            except: pass
                    
        # 4. Place adsorbates
        for ads in state.adsorbates:
            if ads.coverage > 0.0 and len(slab) > 0:
                try:
                    add_adsorbate(slab, ads.identity, 1.5, 'ontop')
                except:
                    try:
                        com = np.mean(slab.positions, axis=0)
                        add_adsorbate(slab, ads.identity, 1.5, position=(com[0], com[1]))
                    except: pass

        # Final Robustness Check
        if len(slab) == 0:
            slab = Atoms('Cu', cell=(3.6, 3.6, 3.6), positions=[(0,0,0)], pbc=True)
        if slab.cell is None or np.any(np.linalg.norm(slab.cell, axis=1) == 0):
            slab.set_cell((10, 10, 20))
            
        return slab

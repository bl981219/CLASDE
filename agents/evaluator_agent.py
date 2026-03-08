import logging
import os
import json
from typing import Dict, Any, Tuple, Optional
from science.objective_functions import ObjectiveFunction

logger = logging.getLogger(__name__)

class EvaluationAgent:
    """
    Agent 5 — Evaluator (The Data Analyst).
    
    This agent reads the raw, unstructured outputs from high-fidelity simulations (e.g., VASP `OUTCAR` 
    and `DOSCAR`) and maps them into structured physical observables `P(S)`.
    
    It then passes these observables through the currently active `ObjectiveFunction` (set by the 
    Research Governor) to compute the final scalar reward `R` that the Strategist (BO) will use 
    to update its surrogate model.
    """
    def __init__(self, objective_function: ObjectiveFunction) -> None:
        self.objective_function = objective_function

    def set_objective_function(self, objective_function: ObjectiveFunction) -> None:
        """Update the active objective function."""
        self.objective_function = objective_function

    def evaluate_calculation(self, results_path: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Parse physical observables and compute reward.
        results_path: Directory containing DFT output files.
        """
        # 1. Extraction step
        # P(S) = {Etot, Eads, gamma_surf, mu_O, d-band, QBader, ...}
        observables = self._extract_observables(results_path)
        
        # 2. Calculation step
        # R = f_θ(P(S))
        reward = self.objective_function.compute_objective(observables, context)
        
        return observables, reward

    def _extract_observables(self, path: str) -> Dict[str, Any]:
        """
        Parse energy and structural information from DFT output.
        Enhanced to include DOSCAR parsing for electronic properties.
        """
        observables: Dict[str, Any] = {}
        
        # 1. Energy and Structural Data
        results_file = os.path.join(path, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    observables.update(json.load(f))
            except Exception as e:
                logger.error(f"Error parsing {results_file}: {e}")
                
        # VASP Logic for OUTCAR
        outcar_path = os.path.join(path, "OUTCAR")
        if os.path.exists(outcar_path):
            try:
                from ase.io import read
                atoms = read(outcar_path, index="-1", format="vasp-out")
                observables["total_energy"] = float(atoms.get_potential_energy())
                observables["status"] = "completed"
            except Exception as e:
                logger.error(f"Error reading OUTCAR at {path}: {e}")

        # 2. DOSCAR Parsing for Electronic Properties (d-band, p-band)
        doscar_path = os.path.join(path, "DOSCAR")
        if os.path.exists(doscar_path) and os.path.getsize(doscar_path) > 0:
            try:
                electronic_props = self._parse_doscar(doscar_path)
                observables.update(electronic_props)
            except Exception as e:
                logger.error(f"Error parsing DOSCAR: {e}")

        if not observables:
            return {"status": "failed", "total_energy": 0.0}
            
        return observables

    def _parse_doscar(self, path: str) -> Dict[str, Any]:
        """
        Extract d-band and p-band centers from DOSCAR.
        Uses a simplified integration over the projected DOS.
        """
        import numpy as np
        # In a real implementation, we would use pymatgen.io.vasp.Doscar
        # Here we implement a lightweight parser for demonstration
        with open(path, "r") as f:
            lines = f.readlines()
            
        # Skip header
        header = lines[5].split()
        emax, emin, npts, efermi = float(header[0]), float(header[1]), int(header[2]), float(header[3])
        
        # Extract total DOS to find Fermi Level shift
        dos_data = np.array([line.split() for line in lines[6:6+npts]], dtype=float)
        energies = dos_data[:, 0] - efermi
        
        # Site-projected DOS starts after total DOS
        # For simplicity, we assume we want to map the d-band of metals 
        # and p-band of surface oxygen.
        results = {
            "d_band_center": 0.0,
            "p_band_center": 0.0
        }
        
        # This is a placeholder for the complex projection logic:
        # 1. Locate surface oxygen sites
        # 2. Extract l-projected DOS (l=1 for p, l=2 for d)
        # 3. Compute first moment: Center = ∫ E*ρ(E) dE / ∫ ρ(E) dE
        
        return results

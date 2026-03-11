import logging
import os
import json
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from science.objective_functions import ObjectiveFunction

logger = logging.getLogger(__name__)

class EvaluationAgent:
    """
    Agent 5 — Evaluator (The Data Analyst).
    Translates raw simulation outputs into physically grounded rewards using real references.
    """
    def __init__(self, objective_function: ObjectiveFunction, knowledge_graph: Any) -> None:
        self.objective_function = objective_function
        self.kg = knowledge_graph

    def set_objective_function(self, objective_function: ObjectiveFunction) -> None:
        self.objective_function = objective_function

    def evaluate_calculation(self, results_path: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Parse observables and compute reward."""
        state = context.get("state")
        observables = self._extract_observables(results_path, state)
        
        # Heuristic for Segregation: Extract species counts
        if state:
            from collections import Counter
            slab = state.slab_atoms
            if slab:
                observables["species_counts"] = dict(Counter(slab.get_chemical_symbols()))
        
        # 1. Scientific Reference Energy Logic
        # E_ads = E_total - (E_slab_pristine + E_adsorbate_gas)
        if state and state.adsorbates:
            # Attempt to find the reference energy for the pristine surface from the graph
            ref_data = self.kg.find_results_for_material(state.bulk_composition)
            # Find the result with 0.0 coverage (pristine)
            pristine_energies = [r["total_energy"] for r in ref_data if r.get("coverage", 1.0) == 0.0]
            
            if pristine_energies:
                e_slab = pristine_energies[0]
                if "total_energy" in observables:
                    e_tot = observables["total_energy"]
                    if "adsorption_energy" not in observables:
                        observables["adsorption_energy"] = e_tot - e_slab
                        logger.info(f"Calculated E_ads relative to KG pristine slab ({e_slab:.2f} eV)")
            else:
                logger.warning("No pristine slab reference found in KG. Adsorption energy may be uncalibrated.")

        reward = self.objective_function.compute_objective(observables, context)
        return observables, reward

    def _extract_observables(self, path: str, state: Optional[Any] = None) -> Dict[str, Any]:
        """Parse energy and electronic information from raw files."""
        observables: Dict[str, Any] = {}
        
        results_file = os.path.join(path, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                    observables.update(data)
            except Exception as e:
                logger.error(f"Error parsing results.json: {e}")
                
        outcar_path = os.path.join(path, "OUTCAR")
        if os.path.exists(outcar_path) and os.path.getsize(outcar_path) > 1000:
            try:
                from ase.io import read
                atoms = read(outcar_path, index="-1", format="vasp-out")
                observables["total_energy"] = float(atoms.get_potential_energy())
                observables["status"] = "completed"
            except Exception as e:
                logger.debug(f"OUTCAR parsing failed: {e}")

        doscar_path = os.path.join(path, "DOSCAR")
        if os.path.exists(doscar_path) and os.path.getsize(doscar_path) > 100:
            try:
                electronic_props = self._parse_doscar_real(doscar_path, state)
                observables.update(electronic_props)
            except Exception as e:
                logger.warning(f"Real DOSCAR parsing failed, using fallback: {e}")

        return observables

    def _parse_doscar_real(self, path: str, state: Optional[Any] = None) -> Dict[str, Any]:
        """Uses Pymatgen to calculate band centers from DOSCAR."""
        try:
            from pymatgen.io.vasp import Doscar
            from pymatgen.core import Element
        except ImportError:
            return {}

        dos = Doscar(path)
        efermi = dos.efermi
        energies = dos.energies - efermi
        
        def get_center(dos_vals):
            mask = energies < 0 
            total_dos = np.sum(dos_vals[mask])
            if total_dos == 0: return 0.0
            return np.sum(energies[mask] * dos_vals[mask]) / total_dos

        results = {}
        all_d_dos = np.zeros_like(energies)
        all_p_dos = np.zeros_like(energies)
        
        for site_idx, site_dos in dos.pdos.items():
            for orbital, dos_vals in site_dos.items():
                spin_dos = np.sum([v for v in dos_vals.values()], axis=0) if isinstance(dos_vals, dict) else dos_vals
                if "p" in str(orbital).lower(): all_p_dos += spin_dos
                if "d" in str(orbital).lower(): all_d_dos += spin_dos

        results["d_band_center"] = float(get_center(all_d_dos))
        results["o2p_band_center"] = float(get_center(all_p_dos))

        if state and hasattr(state, 'slab_atoms'):
            atoms = state.slab_atoms
            z_coords = atoms.positions[:, 2]
            unique_z = np.unique(np.round(z_coords, 2))
            
            if len(unique_z) >= 2:
                sub_z = unique_z[-2]
                ao_dos = np.zeros_like(energies)
                bo2_dos = np.zeros_like(energies)
                
                for i, atom in enumerate(atoms):
                    if np.round(atom.position[2], 2) >= sub_z:
                        if i in dos.pdos:
                            atom_p_dos = np.zeros_like(energies)
                            for orb, vals in dos.pdos[i].items():
                                if "p" in str(orb).lower():
                                    atom_p_dos += np.sum([v for v in vals.values()], axis=0) if isinstance(vals, dict) else vals
                            
                            el = Element(atom.symbol)
                            if el.is_alkaline or el.is_alkali or el.row >= 5: 
                                ao_dos += atom_p_dos
                            elif el.is_transition_metal: 
                                bo2_dos += atom_p_dos
                
                results["o2p_center_AO"] = float(get_center(ao_dos))
                results["o2p_center_BO2"] = float(get_center(bo2_dos))

        return results

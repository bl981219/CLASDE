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
                observables["structure"] = atoms
                observables["status"] = "completed"
            except Exception as e:
                logger.debug(f"OUTCAR parsing failed: {e}")

        # Use Vasprun for electronic properties (more robust than DOSCAR)
        vasprun_path = os.path.join(path, "vasprun.xml")
        if os.path.exists(vasprun_path) and os.path.getsize(vasprun_path) > 1000:
            try:
                electronic_props = self._parse_vasprun_electronic(vasprun_path, state)
                observables.update(electronic_props)
            except Exception as e:
                logger.warning(f"Vasprun parsing failed: {e}")

        return observables

    def _parse_vasprun_electronic(self, path: str, state: Optional[Any] = None) -> Dict[str, Any]:
        """Uses Pymatgen Vasprun to calculate band centers."""
        try:
            from pymatgen.io.vasp import Vasprun
            from pymatgen.core import Element
        except ImportError:
            return {}

        v = Vasprun(path, parse_dos=True, parse_eigen=False)
        dos = v.complete_dos
        if not dos:
            return {}
            
        efermi = dos.efermi
        
        def get_center(dos_obj):
            energies = dos_obj.energies - efermi
            dos_vals = dos_obj.get_densities()
            # Only consider states below Fermi level
            mask = energies < 0 
            total_dos = np.sum(dos_vals[mask])
            if total_dos == 0: return 0.0
            return np.sum(energies[mask] * dos_vals[mask]) / total_dos

        results = {}
        results["d_band_center"] = 0.0 # Placeholder for total d
        results["o2p_band_center"] = 0.0 # Placeholder for total p

        if state and hasattr(state, 'slab_atoms') and state.slab_atoms:
            atoms = state.slab_atoms
            z_coords = atoms.positions[:, 2]
            unique_z = np.unique(np.round(z_coords, 2))
            
            if len(unique_z) >= 2:
                sub_z = unique_z[-2]
                ao_dos_list = []
                bo2_dos_list = []
                
                for i, atom in enumerate(atoms):
                    if np.round(atom.position[2], 2) >= sub_z:
                        # Use get_site_orbital_dos from CompleteDos
                        # It returns a Dict[Orbital, Dos]
                        orb_dos = dos.get_site_orbital_dos(v.final_structure[i])
                        p_vals = np.zeros_like(dos.energies)
                        for orb, pdos_obj in orb_dos.items():
                            if "p" in str(orb).lower():
                                p_vals += pdos_obj.get_densities()
                        
                        el = Element(atom.symbol)
                        if el.is_alkaline or el.is_alkali or el.row >= 5: 
                            ao_dos_list.append(p_vals)
                        elif el.is_transition_metal or (atom.symbol == "O" and np.round(atom.position[2], 2) > sub_z + 0.5):
                            # Heuristic: link O to BO2 if it's in top layer near B
                            bo2_dos_list.append(p_vals)
                
                if ao_dos_list:
                    ao_sum = np.sum(ao_dos_list, axis=0)
                    energies = dos.energies - efermi
                    mask = energies < 0
                    results["o2p_center_AO"] = float(np.sum(energies[mask] * ao_sum[mask]) / np.sum(ao_sum[mask]))
                
                if bo2_dos_list:
                    bo2_sum = np.sum(bo2_dos_list, axis=0)
                    energies = dos.energies - efermi
                    mask = energies < 0
                    results["o2p_center_BO2"] = float(np.sum(energies[mask] * bo2_sum[mask]) / np.sum(bo2_sum[mask]))

        return results

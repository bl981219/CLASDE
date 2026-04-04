import logging
import os
import json
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import yaml
from science.objective_functions import ObjectiveFunction
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class EvaluationAgent:
    """
    Agent 5 — Evaluator (The Data Analyst).
    
    Translates raw simulation outputs into physically grounded rewards using real references.
    """
    def __init__(self, objective_function: ObjectiveFunction, knowledge_graph: Any) -> None:
        self.objective_function = objective_function
        self.kg = knowledge_graph
        self.reference_data = self._load_reference_data()

    def _load_reference_data(self) -> Dict[str, Any]:
        """Loads NIST and standard baseline data from config."""
        ref_path = "configs/reference_data.yaml"
        if os.path.exists(ref_path):
            try:
                with open(ref_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load reference data from {ref_path}: {e}")
        return {"gas_phase": {}}

    def set_objective_function(self, objective_function: ObjectiveFunction) -> None:
        self.objective_function = objective_function

    def evaluate_calculation(self, results_path: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        state: Optional[SurfaceState] = context.get("state")
        observables = self._extract_observables(results_path, state)
        
        is_valid, reason = self.validate_results(observables, context)
        if not is_valid:
            logger.warning(f"[Evaluator] Result validation failed: {reason}")
            return observables, -10.0 

        if state:
            from collections import Counter
            slab = state.slab_structure
            if slab:
                observables["species_counts"] = dict(Counter([s.specie.symbol for s in slab]))
        
        if state and state.adsorbates:
            ref_data = self.kg.find_results_for_material(state.bulk_composition)
            pristine_energies = [r.get("total_energy") for r in ref_data if r.get("coverage", 1.0) == 0.0]
            pristine_energies = [e for e in pristine_energies if e is not None]
            
            gas_refs = self.reference_data.get("gas_phase", {})
            e_gas_total = sum([gas_refs.get(ads.identity, 0.0) for ads in state.adsorbates])

            if pristine_energies:
                e_slab = pristine_energies[0]
                if "total_energy" in observables:
                    e_tot = observables["total_energy"]
                    if "adsorption_energy" not in observables:
                        observables["adsorption_energy"] = e_tot - e_slab - e_gas_total
                        logger.info(f"Calculated E_ads: {observables['adsorption_energy']:.2f} eV")
            else:
                logger.warning("No pristine slab reference found.")
                if "total_energy" in observables and "adsorption_energy" not in observables:
                    observables["adsorption_energy"] = 0.0

        reward = self.objective_function.evaluate(state, observables)
        
        if observables.get("converged") is False:
            logger.info(f"[Evaluator] Applying non-convergence penalty to reward")
            reward -= 0.5

        return observables, reward

    def validate_results(self, observables: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        energy = observables.get("total_energy")
        if energy is None or energy == 0.0:
            return False, "Total energy is missing or exactly zero."
            
        struct = observables.get("structure")
        if struct:
            if isinstance(struct, dict):
                from pymatgen.core import Structure
                try:
                    struct = Structure.from_dict(struct)
                except Exception as e:
                    return False, f"Failed to reconstruct structure: {e}"

            try:
                min_dist = struct.get_all_distances().min()
                if min_dist < 0.5:
                    return False, f"Unphysically short bond: {min_dist:.2f} A"
            except Exception as e:
                logger.debug(f"Structure validation failed: {e}")

        return True, "Success"

    def _extract_observables(self, path: str, state: Optional[SurfaceState] = None) -> Dict[str, Any]:
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
                from pymatgen.io.ase import AseAtomsAdaptor
                atoms = read(outcar_path, index="-1", format="vasp-out")
                observables["total_energy"] = float(atoms.get_potential_energy())
                observables["structure"] = AseAtomsAdaptor.get_structure(atoms)
                observables["status"] = "completed"
            except Exception as e:
                logger.debug(f"OUTCAR parsing failed: {e}")

        vasprun_path = os.path.join(path, "vasprun.xml")
        if os.path.exists(vasprun_path) and os.path.getsize(vasprun_path) > 1000:
            try:
                electronic_props = self._parse_vasprun_electronic(vasprun_path, state)
                observables.update(electronic_props)
            except Exception as e:
                logger.warning(f"Vasprun parsing failed: {e}")

        return observables

    def _parse_vasprun_electronic(self, path: str, state: Optional[SurfaceState] = None) -> Dict[str, Any]:
        try:
            from pymatgen.io.vasp import Vasprun
            from science.descriptors import SurfaceDescriptors
        except ImportError:
            return {}

        try:
            v = Vasprun(path, parse_dos=True, parse_eigen=False)
            dos = v.complete_dos
            if not dos:
                return {}
                
            efermi = dos.efermi
            energies = dos.energies - efermi
            
            results = {}
            
            if state and hasattr(state, 'slab_structure') and state.slab_structure:
                from science.chemistry import ChemistryPhysicist
                struct = state.slab_structure
                z_coords = struct.cart_coords[:, 2]
                unique_z = np.unique(np.round(z_coords, 2))
                
                if len(unique_z) >= 2:
                    sub_z = unique_z[-2]
                    ao_dos_list = []
                    bo2_dos_list = []
                    
                    for i, site in enumerate(struct):
                        if np.round(site.coords[2], 2) >= sub_z:
                            orb_dos = dos.get_site_orbital_dos(site)
                            p_vals = np.zeros_like(dos.energies)
                            for orb, pdos_obj in orb_dos.items():
                                if "p" in str(orb).lower():
                                    p_vals += pdos_obj.get_densities()
                            
                            symbol = site.specie.symbol
                            if ChemistryPhysicist.get_layer_type([symbol]) == "AO":
                                ao_dos_list.append(p_vals)
                            else:
                                bo2_dos_list.append(p_vals)
                    
                    if ao_dos_list:
                        ao_sum = np.sum(ao_dos_list, axis=0)
                        results["o2p_center_AO"] = SurfaceDescriptors.calculate_o2p_band_center(energies, ao_sum)
                    
                    if bo2_dos_list:
                        bo2_sum = np.sum(bo2_dos_list, axis=0)
                        results["o2p_center_BO2"] = SurfaceDescriptors.calculate_o2p_band_center(energies, bo2_sum)

            return results
        except Exception as e:
            logger.error(f"Electronic parsing failed: {e}")
            return {}

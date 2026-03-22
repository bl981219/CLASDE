import logging
import os
import json
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from science.objective_functions import ObjectiveFunction
from core.state import SurfaceState

logger = logging.getLogger(__name__)

class EvaluationAgent:
    """
    Agent 5 — Evaluator (The Data Analyst).
    
    Translates raw simulation outputs into physically grounded rewards using real references.
    It parses electronic structure files (like vasprun.xml) to extract descriptors 
    like the O 2p-band center, and calculates adsorption energies relative to pristine surfaces.
    """
    def __init__(self, objective_function: ObjectiveFunction, knowledge_graph: Any) -> None:
        """
        Initializes the Evaluator.

        Args:
            objective_function (ObjectiveFunction): The mathematical function used to calculate the reward.
            knowledge_graph (Any): The semantic graph storing previous experimental results.
        """
        self.objective_function = objective_function
        self.kg = knowledge_graph
        self.reference_data = self._load_reference_data()

    def _load_reference_data(self) -> Dict[str, Any]:
        """Loads NIST and standard baseline data from config."""
        import yaml
        ref_path = "configs/reference_data.yaml"
        if os.path.exists(ref_path):
            try:
                with open(ref_path, "r") as f:
                    return yaml.safe_load(f)
            except:
                pass
        return {"gas_phase": {}}

    def set_objective_function(self, objective_function: ObjectiveFunction) -> None:
        """Dynamically update the objective function."""
        self.objective_function = objective_function

    def evaluate_calculation(self, results_path: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Parse observables from a calculation directory and compute the formal reward.

        Args:
            results_path (str): Path to the directory containing calculation outputs.
            context (Dict[str, Any]): Contextual information, crucially including the 'state' (SurfaceState).

        Returns:
            Tuple[Dict[str, Any], float]: A tuple containing the parsed observables and the calculated reward.
        """
        state: Optional[SurfaceState] = context.get("state")
        observables = self._extract_observables(results_path, state)
        
        # 1. Scientific Sanity Check (New)
        is_valid, reason = self.validate_results(observables, context)
        if not is_valid:
            logger.warning(f"[Evaluator] Result validation failed: {reason}")
            # Penalty for unphysical results
            return observables, -10.0 

        # 2. Heuristic for Segregation: Extract species counts
        if state:
            from collections import Counter
            slab = state.slab_structure
            if slab:
                observables["species_counts"] = dict(Counter([s.specie.symbol for s in slab]))
        
        # 3. Scientific Reference Energy Logic
        # E_ads = E_total - (E_slab_pristine + E_adsorbate_gas)
        if state and state.adsorbates:
            # 1.1 Reference Energy for Pristine Slab
            ref_data = self.kg.find_results_for_material(state.bulk_composition)
            # Safe access with .get()
            pristine_energies = [r.get("total_energy") for r in ref_data if r.get("coverage", 1.0) == 0.0]
            pristine_energies = [e for e in pristine_energies if e is not None]
            
            # 1.2 Reference Energy for Gas-phase Adsorbate (Standard Values)
            gas_refs = self.reference_data.get("gas_phase", {})
            e_gas_total = sum([gas_refs.get(ads.identity, 0.0) for ads in state.adsorbates])

            if pristine_energies:
                e_slab = pristine_energies[0]
                if "total_energy" in observables:
                    e_tot = observables["total_energy"]
                    if "adsorption_energy" not in observables:
                        # Simple adsorption energy calculation
                        observables["adsorption_energy"] = e_tot - e_slab - e_gas_total
                        logger.info(f"Calculated E_ads: {observables['adsorption_energy']:.2f} eV (Ref: {e_slab:.2f} eV, Gas: {e_gas_total:.2f} eV)")
            else:
                # If no reference exists, we use the current total energy as a relative marker 
                # to prevent campaign death, but warn the user.
                logger.warning("No pristine slab reference found. Using uncalibrated total energy as temporary reward.")
                if "total_energy" in observables and "adsorption_energy" not in observables:
                    observables["adsorption_energy"] = 0.0 # Neutralize E_ads to allow loop to continue

        reward = self.objective_function.compute_objective(observables, context)
        return observables, reward

    def validate_results(self, observables: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Performs physical sanity checks on the calculation results.
        Checks for non-zero energy and reasonable structural features.
        """
        energy = observables.get("total_energy")
        if energy is None or energy == 0.0:
            return False, "Total energy is missing or exactly zero (likely convergence failure)."
            
        # Check for unphysical structure if it exists
        struct = observables.get("structure")
        if struct:
            # Handle dictionary form if reloaded from JSON
            if isinstance(struct, dict):
                from pymatgen.core import Structure
                try:
                    struct = Structure.from_dict(struct)
                except:
                    return False, "Failed to reconstruct structure from data."

            # Simple check for minimum bond distance
            try:
                min_dist = struct.get_all_distances().min()
                if min_dist < 0.5: # Angstrom
                    return False, f"Unphysically short bond detected: {min_dist:.2f} A"
            except Exception as e:
                logger.debug(f"Structure validation failed during distance check: {e}")

        return True, "Success"

    def _extract_observables(self, path: str, state: Optional[SurfaceState] = None) -> Dict[str, Any]:
        """
        Parse energy and electronic information from raw files in the specified path.

        Args:
            path (str): The directory containing the output files.
            state (Optional[SurfaceState]): The physical state associated with the calculation.

        Returns:
            Dict[str, Any]: A dictionary of extracted observables (e.g., total_energy, structure).
        """
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
                # Fix: Always return Pymatgen Structure in observables
                observables["structure"] = AseAtomsAdaptor.get_structure(atoms)
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

    def _parse_vasprun_electronic(self, path: str, state: Optional[SurfaceState] = None) -> Dict[str, Any]:
        """
        Uses Pymatgen Vasprun to calculate layer-resolved band centers.

        Args:
            path (str): Path to the vasprun.xml file.
            state (Optional[SurfaceState]): The physical state containing the slab structure.

        Returns:
            Dict[str, Any]: A dictionary containing calculated electronic properties.
        """
        try:
            from pymatgen.io.vasp import Vasprun
            from pymatgen.core import Element
            from science.descriptors import SurfaceDescriptors
        except ImportError:
            return {}

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
                        # Use get_site_orbital_dos from CompleteDos
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

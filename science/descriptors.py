import numpy as np
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SurfaceDescriptors:
    """
    Computes standard physical and electronic descriptors for surface catalysis.
    """
    @staticmethod
    def compute_coordination_number(atoms: Any, atom_index: int, cutoff: float = 3.0) -> int:
        """Calculate simple coordination number based on distance cutoff."""
        try:
            distances = atoms.get_distances(atom_index, range(len(atoms)), mic=True)
            # Exclude self (distance 0)
            return int(np.sum((distances > 0.1) & (distances <= cutoff)))
        except Exception as e:
            logger.warning(f"Could not compute coordination number: {e}")
            return 0

    @staticmethod
    def compute_gcn(atoms: Any, atom_index: int, cutoff: float = 3.0) -> float:
        """
        Calculates the Generalized Coordination Number (GCN).
        GCN = sum(cn_i / cn_max) for all neighbors i.
        """
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.analysis.local_env import CrystalNN
            
            # 1. Identify neighbors
            distances = atoms.get_distances(atom_index, range(len(atoms)), mic=True)
            neighbor_indices = np.where((distances > 0.1) & (distances <= cutoff))[0]
            
            if len(neighbor_indices) == 0:
                return 0.0
                
            # 2. Determine cn_max dynamically using CrystalNN on the bulk-like environment
            # If we can't find bulk, we default to 12 (fcc) or 8 (bcc) based on chemistry
            struct = AseAtomsAdaptor.get_structure(atoms)
            cnn = CrystalNN()
            # Get CN for a bulk-like atom of the same species
            cn_max = 12.0 # Standard fallback
            try:
                # Find an atom of the same species that is 'deep' in the slab
                z_coords = atoms.positions[:, 2]
                deep_idx = [i for i, z in enumerate(z_coords) if z < np.median(z_coords) and atoms[i].symbol == atoms[atom_index].symbol]
                if deep_idx:
                    cn_max = float(cnn.get_cn(struct, deep_idx[0]))
            except: pass
            
            gcn = 0.0
            for idx in neighbor_indices:
                neighbor_cn = SurfaceDescriptors.compute_coordination_number(atoms, idx, cutoff)
                gcn += neighbor_cn / cn_max
                
            return float(gcn)
        except Exception as e:
            logger.warning(f"Could not compute GCN: {e}")
            return 0.0

    @staticmethod
    def compute_lattice_strain(current_lattice: float, bulk_lattice: float) -> float:
        """Calculate percentage lattice strain."""
        if bulk_lattice == 0: return 0.0
        return float((current_lattice - bulk_lattice) / bulk_lattice)

    @staticmethod
    def extract_d_band_center(dos_data: Dict[str, np.ndarray]) -> float:
        """Calculates d-band center from DOS data."""
        try:
            energies = dos_data['energies']
            d_dos = dos_data['d_dos']
            numerator = np.trapz(energies * d_dos, energies)
            denominator = np.trapz(d_dos, energies)
            return float(numerator / denominator) if denominator != 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to extract d-band center: {e}")
            return 0.0

    @staticmethod
    def extract_d_band_edge(dos_data: Dict[str, np.ndarray]) -> float:
        """
        Identifies the d-band edge energy (highest energy occupied d-state).
        Approximated here as the energy where the integrated d-DOS reaches 90% of total.
        """
        try:
            energies = dos_data['energies']
            d_dos = dos_data['d_dos']
            # Only consider energies below Fermi (assumed at 0 in input)
            mask = energies <= 0
            e_occ = energies[mask]
            dos_occ = d_dos[mask]
            
            cumulative_dos = np.cumsum(dos_occ)
            if len(cumulative_dos) == 0: return 0.0
            
            threshold = 0.9 * cumulative_dos[-1]
            edge_idx = np.where(cumulative_dos >= threshold)[0][0]
            return float(e_occ[edge_idx])
        except Exception as e:
            logger.warning(f"Failed to extract d-band edge: {e}")
            return 0.0

    @staticmethod
    def extract_o2p_band_center(dos_data: Dict[str, np.ndarray]) -> float:
        """Calculates oxygen 2p-band center from DOS data."""
        try:
            energies = dos_data['energies']
            p_dos = dos_data['p_dos']
            numerator = np.trapz(energies * p_dos, energies)
            denominator = np.trapz(p_dos, energies)
            return float(numerator / denominator) if denominator != 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to extract O2p-band center: {e}")
            return 0.0

    @staticmethod
    def calculate_vacancy_formation_energy(e_defective: float, e_pristine: float, e_chemical_potential: float) -> float:
        """
        Calculates the formation energy of a surface vacancy.
        Evac = E_defective - E_pristine + mu_species
        """
        return float(e_defective - e_pristine + e_chemical_potential)

    @staticmethod
    def calculate_charge_transfer_energy(m_d_center: float, o_p_center: float) -> float:
        """Delta = Md - Op"""
        return float(m_d_center - o_p_center)

    @staticmethod
    def extract_work_function(fermi_level: float, vacuum_level: float) -> float:
        """Phi = Evac - Efermi"""
        return float(vacuum_level - fermi_level)

    @staticmethod
    def calculate_eg_occupancy(d_dos: np.ndarray, energies: np.ndarray) -> float:
        """
        Heuristic occupancy of eg orbitals in octahedrally coordinated metals.
        Assumes eg states are the higher energy portion of the d-band.
        """
        # Placeholder: In production, requires orbital-projected DOS (m-projected)
        return 0.0 

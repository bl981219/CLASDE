import numpy as np
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class SurfaceDescriptors:
    """
    Computes standard physical and electronic descriptors for surface catalysis.
    """
    @staticmethod
    def compute_coordination_number(atoms: Any, atom_index: int, cutoff: float = 2.5) -> int:
        """Calculate simple coordination number based on distance cutoff."""
        try:
            distances = atoms.get_distances(atom_index, range(len(atoms)))
            # Exclude self (distance 0)
            return int(np.sum((distances > 0) & (distances <= cutoff)))
        except Exception as e:
            logger.warning(f"Could not compute coordination number: {e}")
            return 0

    @staticmethod
    def extract_d_band_center(dos_data: Dict[str, np.ndarray]) -> float:
        """
        Calculates d-band center from DOS data.
        dos_data expects keys 'energies' and 'd_dos'
        """
        try:
            energies = dos_data['energies']
            d_dos = dos_data['d_dos']
            numerator = np.trapz(energies * d_dos, energies)
            denominator = np.trapz(d_dos, energies)
            if denominator == 0:
                return 0.0
            return float(numerator / denominator)
        except Exception as e:
            logger.warning(f"Failed to extract d-band center: {e}")
            return 0.0

    @staticmethod
    def parse_bader_charges(bader_output_file: str) -> List[float]:
        """Parses Bader charge analysis output."""
        charges = []
        try:
            with open(bader_output_file, 'r') as f:
                lines = f.readlines()
                # Dummy parse: assume standard ACF.dat format starting line 3
                for line in lines[2:]:
                    parts = line.split()
                    if len(parts) >= 5:
                        charges.append(float(parts[4]))
        except Exception as e:
            logger.warning(f"Failed to parse Bader charges: {e}")
        return charges

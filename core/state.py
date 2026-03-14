import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
import hashlib
import json

logger = logging.getLogger(__name__)

class AdsorbateInstance(BaseModel):
    """Specific instance of an adsorbate on a surface site."""
    identity: str = Field(..., description="Chemical formula of the adsorbate, e.g., 'CO'")
    site_type: str = Field("top", description="Site type, e.g., 'top', 'bridge', 'fcc_hollow', 'hcp_hollow'")
    site_coordination: Optional[int] = Field(None, description="Coordination number of the adsorption site")
    site_neighbor_atoms: List[str] = Field(default_factory=list, description="Elements of neighboring surface atoms")
    coverage: float = Field(0.0, description="Coverage fraction of this specific adsorbate")
    orientation: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Orientation descriptors (e.g., tilt angle, azimuthal)")
    geometry: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Local adsorption geometry (bond lengths, etc.)")

class SurfaceState(BaseModel):
    """
    Formal canonical representation of the atomistic surface configuration space.
    
    This object acts as the universal "Source of Truth" across all agents. 
    It is designed to be physically expressive for surface science, containing 
    bulk properties, surface specifics, and detailed adsorbate configurations.
    """
    bulk_composition: Dict[str, float] = Field(..., description="Bulk composition vector c")
    miller_index: Tuple[int, int, int] = Field(..., description="Miller index (h, k, l)")
    termination: str = Field(..., description="Surface termination descriptor τ")
    
    slab_structure: Optional[Any] = Field(None, description="Physical structure object (Pymatgen Structure). Excluded from direct hashing.")
    
    def get_ase_atoms(self) -> Any:
        """Helper to convert internal Pymatgen structure to ASE Atoms for calculators."""
        if self.slab_structure is None: return None
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            # Check if slab_structure is ASE Atoms or Pymatgen Structure
            if hasattr(self.slab_structure, "get_potential_energy"):
                return self.slab_structure
            return AseAtomsAdaptor.get_atoms(self.slab_structure)
        except: return None
    
    available_sites: List[Dict[str, Any]] = Field(default_factory=list, description="Available adsorption sites on this surface")
    adsorbates: List[AdsorbateInstance] = Field(default_factory=list, description="List of adsorbates on the surface")
    coverage: float = Field(0.0, description="Total coverage θ (legacy/summary field)")
    
    defects: List[Dict] = Field(default_factory=list, description="Defect vector d")
    strain: Tuple[float, float, float] = Field((0.0, 0.0, 0.0), description="Strain applied to the slab (xx, yy, xy)")
    
    temperature: float = Field(298.15, description="Temperature in Kelvin")
    pressure: float = Field(1.0, description="Pressure in atm")
    external_conditions: Dict[str, float] = Field(
        default_factory=lambda: {"Phi": 0.0},
        description="Other external conditions (e.g., Electrochemical Potential Φ)"
    )
    
    metadata: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def to_json(self) -> str:
        """Serialize state to a canonical JSON string."""
        return json.dumps(self.model_dump(exclude={'slab_structure', 'available_sites'}), sort_keys=True)

    def get_id(self) -> str:
        """Generate a unique SHA-256 hash for this state."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def get_feature_vector(self) -> List[float]:
        """
        Convert the structured state into a numerical feature vector.
        V3: Universal encoding using periodic table indices (1-100).
        """
        bulk_vector = [0.0] * 100
        from ase.data import atomic_numbers
        for el, fraction in self.bulk_composition.items():
            z = atomic_numbers.get(el)
            if z and z <= 100:
                bulk_vector[z-1] = float(fraction)
        
        miller_feats = []
        for i in self.miller_index:
            miller_feats.extend([float(i), float(i)**2])
            
        ads_sum = 0.0
        if self.adsorbates:
            from ase import Atoms
            try:
                primary_ads = self.adsorbates[0].identity
                temp_atoms = Atoms(primary_ads)
                ads_sum = float(sum(temp_atoms.get_atomic_numbers()))
            except:
                ads_sum = -1.0
        ads_feat = [ads_sum]
        
        cond_feats = [
            self.coverage,
            self.temperature / 1000.0,
            self.pressure,
            self.external_conditions.get("Phi", 0.0)
        ]
        
        v_count = sum(1 for d in self.defects if d.get("type") == "vacancy")
        s_count = sum(1 for d in self.defects if d.get("type") == "substitution")
        defect_feats = [float(v_count), float(s_count)]
        
        return bulk_vector + miller_feats + ads_feat + cond_feats + defect_feats

    def get_summary(self) -> str:
        """Generate a human-readable summary string for folder naming."""
        comp = "".join([f"{k}{v}" for k, v in self.bulk_composition.items() if v > 0])
        facet = "".join([str(i) for i in self.miller_index])
        
        defect_str = ""
        if self.defects:
            last = self.defects[-1]
            if last.get("type") == "vacancy":
                defect_str = f"_vac_{last.get('site')}"
            elif last.get("type") == "substitution":
                defect_str = f"_sub_{last.get('dopant')}"
        
        ads_str = f"_ads_{self.adsorbates[0].identity}" if self.adsorbates else ""
        return f"{comp}_{facet}_{self.termination}{defect_str}{ads_str}"

    def __hash__(self) -> int:
        return int(self.get_id(), 16)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SurfaceState):
            return False
        return self.get_id() == other.get_id()

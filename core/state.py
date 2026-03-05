from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import hashlib
import json

class SurfaceState(BaseModel):
    """
    Formal representation of the atomistic surface configuration space.
    S = {c, τ, (h,k,l), d, a, θ, (T, p, Φ)}
    """
    bulk_composition: Dict[str, float] = Field(..., description="Bulk composition vector c")
    miller_index: Tuple[int, int, int] = Field(..., description="Miller index (h, k, l)")
    termination: str = Field(..., description="Surface termination descriptor τ")
    defects: List[Dict] = Field(default_factory=list, description="Defect vector d")
    adsorbate: Optional[str] = Field(None, description="Adsorbate identity a")
    coverage: float = Field(0.0, description="Coverage θ")
    external_conditions: Dict[str, float] = Field(
        default_factory=lambda: {"T": 298.15, "p": 1.0, "Phi": 0.0},
        description="External conditions (T, p, Φ)"
    )
    
    # Metadata for tracking
    metadata: Dict = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize state to a canonical JSON string."""
        return json.dumps(self.model_dump(), sort_keys=True)

    def get_id(self) -> str:
        """Generate a unique SHA-256 hash for this state."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def __hash__(self):
        return int(self.get_id(), 16)

    def __eq__(self, other):
        if not isinstance(other, SurfaceState):
            return False
        return self.get_id() == other.get_id()

    @property
    def feature_vector(self) -> List[float]:
        """
        Convert the structured state into a numerical feature vector.
        V2: Stoichiometry + Descriptor-based encoding.
        """
        # 1. Stoichiometry of bulk (normalized using a fixed chemical space)
        # Assuming the search space includes La, Sr, Mn, O
        chem_space = ["La", "Sr", "Mn", "O"]
        bulk_stoich = []
        for el in chem_space:
            bulk_stoich.append(float(self.bulk_composition.get(el, 0.0)))
        
        # 2. Miller index encoding
        miller_feats = []
        for i in self.miller_index:
            miller_feats.extend([float(i), float(i)**2])
            
        # 3. Adsorbate encoding (One-hot or atomic number)
        # Mock: Simple mapping for common adsorbates
        adsorbate_map = {"O": 8, "OH": 9, "H2O": 10, "CO": 14, None: 0}
        ads_feat = [float(adsorbate_map.get(self.adsorbate, -1))]
        
        # 4. Coverage and External conditions
        cond_feats = [
            self.coverage,
            self.external_conditions.get("T", 298.15) / 1000.0,
            self.external_conditions.get("p", 1.0),
            self.external_conditions.get("Phi", 0.0)
        ]
        
        # 5. Defect fingerprint (Count by type)
        v_count = sum(1 for d in self.defects if d.get("type") == "vacancy")
        s_count = sum(1 for d in self.defects if d.get("type") == "substitution")
        defect_feats = [float(v_count), float(s_count)]
        
        return bulk_stoich + miller_feats + ads_feat + cond_feats + defect_feats

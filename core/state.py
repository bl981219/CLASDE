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
        Simplified version for CLASDE v1.
        """
        # Encode bulk composition (e.g., fractional content)
        # Assuming binary oxide [A: x, B: 1-x]
        bulk_feats = list(self.bulk_composition.values())
        
        # Miller index (normalized)
        miller_feats = [float(i) for i in self.miller_index]
        
        # Coverage
        cov_feat = [self.coverage]
        
        # External conditions
        cond_feats = [
            self.external_conditions.get("T", 298.15) / 1000.0,
            self.external_conditions.get("p", 1.0),
            self.external_conditions.get("Phi", 0.0)
        ]
        
        # Defect count (simple feature)
        defect_feat = [float(len(self.defects))]
        
        # Combined vector
        return bulk_feats + miller_feats + cov_feat + cond_feats + defect_feat

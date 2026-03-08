import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Any
import hashlib
import json

logger = logging.getLogger(__name__)

class SurfaceState(BaseModel):
    """
    Formal canonical representation of the atomistic surface configuration space.
    
    This object acts as the universal "Source of Truth" across all agents. Rather than 
    passing heavy 3D atomic structures (like ASE Atoms objects) between agents, CLASDE 
    passes this lightweight mathematical descriptor. 
    
    S = {c, τ, (h,k,l), d, a, θ, (T, p, Φ)}
    
    Attributes:
        bulk_composition (c): Stoichiometry of the host material (e.g., {"La": 0.5, "Sr": 0.5, "Mn": 1.0, "O": 3.0}).
        miller_index (h,k,l): Crystallographic plane of the surface.
        termination (τ): Descriptor for the topmost layer (e.g., "SrO" or "MnO2").
        defects (d): Vector of point defects, like vacancies or substitutions.
        adsorbate (a): Identity of the bound molecule/atom (e.g., "OH").
        coverage (θ): Fractional surface coverage of the adsorbate.
        external_conditions: Environmental parameters: Temperature (T), Pressure (p), Electrochemical Potential (Φ).
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
    
    # Metadata for tracking origin, lineage, or physical file paths
    metadata: Dict = Field(default_factory=dict)

    def is_physically_equivalent(self, other: Any) -> bool:
        """
        Check if two states represent the same physical structure using Pymatgen StructureMatcher.
        Useful for identifying symmetry-equivalent surfaces or redundant mutations.
        """
        if not isinstance(other, SurfaceState):
            return False
            
        # Quick check for identical hashes
        if self.get_id() == other.get_id():
            return True
            
        # Detailed check using Pymatgen
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            from agents.builder_agent import StructureBuilder
            
            builder = StructureBuilder()
            s1 = builder.build_structure(self)
            s2 = builder.build_structure(other)
            
            if s1 is None or s2 is None:
                return False
                
            # Convert ASE to Pymatgen
            from pymatgen.io.ase import AseAtomsAdaptor
            pmg1 = AseAtomsAdaptor.get_structure(s1)
            pmg2 = AseAtomsAdaptor.get_structure(s2)
            
            matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)
            return matcher.fit(pmg1, pmg2)
        except ImportError:
            # Fallback to hash equality if tools are missing
            return self.get_id() == other.get_id()
        except Exception as e:
            logger.error(f"Error checking physical equivalence: {e}")
            return False

    def to_json(self) -> str:
        """Serialize state to a canonical JSON string."""
        return json.dumps(self.model_dump(), sort_keys=True)

    def get_summary(self) -> str:
        """Generate a human-readable summary string for folder naming."""
        comp = "".join([f"{k}{v}" for k, v in self.bulk_composition.items() if v > 0])
        facet = "".join([str(i) for i in self.miller_index])
        
        # Identify the most recent defect or adsorbate
        defect_str = ""
        if self.defects:
            last = self.defects[-1]
            if last.get("type") == "vacancy":
                defect_str = f"_vac_{last.get('site')}"
            elif last.get("type") == "substitution":
                defect_str = f"_sub_{last.get('dopant')}"
        
        ads_str = f"_ads_{self.adsorbate}" if self.adsorbate else ""
        
        return f"{comp}_{facet}{defect_str}{ads_str}"

    def get_id(self) -> str:
        """Generate a unique SHA-256 hash for this state."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def __hash__(self) -> int:
        return int(self.get_id(), 16)

    def __eq__(self, other: Any) -> bool:
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

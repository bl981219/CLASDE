"""
Typed configuration models using Pydantic.

Provides type-safe, validated configuration for campaigns.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Literal


class ObjectiveConfig(BaseModel):
    """Configuration for objective function."""
    type: str
    target_e_ads: Optional[float] = None
    adsorbate: Optional[str] = None
    
    @validator("type")
    def validate_type(cls, v):
        allowed = [
            "stability", "adsorption_tuning", "bandgap",
            "work_function", "vacancy_formation"
        ]
        if v not in allowed:
            # Allow custom objectives via plugin system
            pass
        return v


class BudgetConfig(BaseModel):
    """Configuration for campaign budget."""
    max_evaluations: int = Field(..., gt=0, description="Maximum number of DFT evaluations")
    max_time_hours: Optional[float] = Field(None, gt=0, description="Time limit in hours")
    max_cost_dollars: Optional[float] = Field(None, gt=0, description="Cost limit in dollars")


class ConstraintsConfig(BaseModel):
    """Configuration for material constraints."""
    facet: List[int] = Field(..., description="Miller indices of surface facet")
    bulk: Dict[str, float] = Field(..., description="Bulk composition")
    
    @validator("facet")
    def validate_facet(cls, v):
        if len(v) != 3:
            raise ValueError("Facet must be 3 Miller indices")
        return v


class AcquisitionConfig(BaseModel):
    """Configuration for acquisition function."""
    acquisition_type: Literal["EI", "UCB", "scientific", "PI"] = Field(
        "EI",
        description="Acquisition strategy"
    )
    kappa: float = Field(2.576, ge=0, description="Exploration parameter")
    batch_size: int = Field(1, ge=1, le=20, description="Parallel evaluations")


class VASPConfig(BaseModel):
    """VASP-specific parameters."""
    PREC: str = "Accurate"
    ENCUT: int = Field(450, gt=0)
    ISMEAR: int = 0
    SIGMA: float = Field(0.05, gt=0)
    NSW: int = Field(100, ge=0)
    IBRION: int = 2
    LORBIT: Optional[int] = 11
    
    class Config:
        extra = "allow"  # Allow additional VASP params


class ComputeConfig(BaseModel):
    """Configuration for computational backend."""
    mode: Literal["local_emt", "mlip", "vasp"] = Field(
        "local_emt",
        description="Simulation fidelity"
    )
    backend: Literal["local", "slurm", "kubernetes"] = Field(
        "local",
        description="Execution platform"
    )
    vasp_params: Optional[VASPConfig] = None
    
    # Backend-specific settings
    slurm_partition: Optional[str] = None
    slurm_extra_header: Optional[str] = None
    ntasks: int = Field(1, gt=0)
    nodes: int = Field(1, gt=0)


class OptimizationConfig(BaseModel):
    """Configuration for optimization strategy."""
    batch_size: int = Field(1, ge=1, description="Deprecated: use acquisition.batch_size")


class CampaignConfig(BaseModel):
    """
    Complete validated configuration for a CLASDE campaign.
    
    This provides type-safe access to all campaign parameters
    with automatic validation.
    """
    
    name: str = Field(..., description="Campaign name")
    description: str = Field("", description="Campaign description")
    
    objective: ObjectiveConfig
    budget: BudgetConfig
    constraints: ConstraintsConfig
    
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    optimization: Optional[OptimizationConfig] = None
    
    results_dir: str = Field("data/results", description="Output directory")
    original_prompt: Optional[str] = Field(None, description="Natural language goal")
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility
    
    @validator("results_dir")
    def validate_results_dir(cls, v):
        """Ensure results directory path is clean."""
        return v.rstrip("/")
    
    def get_batch_size(self) -> int:
        """Get batch size from acquisition config (backward compatible)."""
        if self.optimization and self.optimization.batch_size:
            return self.optimization.batch_size
        return self.acquisition.batch_size

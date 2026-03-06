from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from core.action import ActionType

class Campaign(BaseModel):
    """
    Formal representation of a scientific research campaign.
    
    A campaign represents a bounded set of experiments designed to test a specific 
    scientific hypothesis or optimize a specific property within a defined subspace.
    """
    name: str = Field(..., description="A short, descriptive name for the campaign.")
    objective: Dict[str, Any] = Field(..., description="The objective configuration (e.g., target adsorption energy).")
    material_space: List[str] = Field(default_factory=list, description="A list of base materials or structures to explore.")
    action_space: List[ActionType] = Field(default_factory=list, description="Allowed mutation operators for this campaign.")
    budget: int = Field(50, description="The maximum number of evaluations allowed.")
    description: str = Field("", description="A human-readable description of the scientific goal.")

from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ActionType(str, Enum):
    """
    Enumerated discrete mutation operators.
    These define the fundamental "moves" the agent can make in the configuration space.
    By restricting actions to physically meaningful mutations, we avoid generating
    arbitrary, unphysical structures.
    """
    CHANGE_TERMINATION = "change_termination"       # e.g., switching from TiO2- to SrO-terminated SrTiO3
    INTRODUCE_VACANCY = "introduce_vacancy"         # e.g., removing an oxygen atom
    SUBSTITUTIONAL_DOPANT = "substitutional_dopant" # e.g., replacing La with Sr
    CHANGE_ADSORBATE = "change_adsorbate"           # e.g., replacing O* with OH*
    MODIFY_COVERAGE = "modify_coverage"             # e.g., changing from 0.25 ML to 0.5 ML
    ALTER_CHARGE_STATE = "alter_charge_state"       # e.g., adjusting the applied electrochemical potential
    SWAP_ATOMS = "swap_atoms"                       # e.g., swapping a surface La with a bulk Sr

class MutationAction(BaseModel):
    """
    Structured action A representing a state mutation.
    
    This is the formal representation of an experiment modification. The `OptimizationStrategist` 
    proposes these actions, and the `TransitionEngine` applies them to a `SurfaceState`.
    
    Attributes:
        action_type: The categorical type of the mutation (from ActionType).
        parameters: A dictionary containing the specifics (e.g., {"dopant": "Sr", "site": "La"}).
        reasoning: Optional string explaining *why* the agent chose this action (useful for LLM integration).
    """
    action_type: ActionType = Field(..., description="The type of mutation to perform")
    parameters: Dict[str, Any] = Field(..., description="Parameters required for the mutation")
    reasoning: Optional[str] = Field(None, description="Optional reasoning for this action choice")

    def __repr__(self):
        return f"Action({self.action_type.value}, {self.parameters})"

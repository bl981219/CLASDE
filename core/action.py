from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ActionType(str, Enum):
    """Enumerated discrete mutation operators."""
    CHANGE_TERMINATION = "change_termination"
    INTRODUCE_VACANCY = "introduce_vacancy"
    SUBSTITUTIONAL_DOPANT = "substitutional_dopant"
    CHANGE_ADSORBATE = "change_adsorbate"
    MODIFY_COVERAGE = "modify_coverage"
    ALTER_CHARGE_STATE = "alter_charge_state"

class MutationAction(BaseModel):
    """
    Structured action A representing a state mutation.
    """
    action_type: ActionType = Field(..., description="The type of mutation to perform")
    parameters: Dict[str, Any] = Field(..., description="Parameters required for the mutation")
    reasoning: Optional[str] = Field(None, description="Optional reasoning for this action choice")

    def __repr__(self):
        return f"Action({self.action_type.value}, {self.parameters})"

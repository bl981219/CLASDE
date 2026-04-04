from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid
import time

class KnowledgeLevel(int, Enum):
    THEORY = 1
    HYPOTHESIS = 2
    EXPERIMENT = 3
    INSIGHT = 4

class ResearchIdea(BaseModel):
    """Level 1: Theory (PI)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = Field(..., description="High-level research goal")
    intuition: str = Field(..., description="Scientific intuition or rationale")
    constraints: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

class Critique(BaseModel):
    """Scientific Gatekeeping (Postdoc)"""
    idea_id: str
    validity: bool = Field(..., description="Whether the PI's idea is scientifically sound/feasible")
    issues: List[str] = Field(default_factory=list)
    revised_plan: Optional[str] = Field(None, description="Postdoc's suggested correction if validity is False")
    confidence: float = Field(..., ge=0.0, le=1.0)

class Hypothesis(BaseModel):
    """Level 2: Strict Scientific Hypothesis (Postdoc)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idea_id: str
    variable: str = Field(..., description="Controllable physical variable (e.g. 'dopant concentration')")
    manipulation: str = Field(..., description="Proposed change (e.g. 'increase x from 0 to 0.2')")
    expected_effect: str = Field(..., description="Predicted physical outcome")
    metric: str = Field(..., description="Measurable metric (e.g. 'E_ads')")
    falsification_condition: str = Field(..., description="Condition that would disprove this hypothesis")
    test_plan: Dict[str, Any] = Field(default_factory=dict)
    status: str = "untested"
    confidence: float = 0.5

    @validator("metric")
    def metric_must_exist(cls, v):
        allowed = ["E_ads", "stability", "reward", "adsorption_energy", "vacancy_formation_energy"]
        if v not in allowed:
            raise ValueError(f"Metric {v} not in allowed list: {allowed}")
        return v

class Experiment(BaseModel):
    """Level 3: Concrete Simulation (Execution)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str
    parameters: Dict[str, Any]
    method: str # DFT, MLIP, etc.
    expected_output: List[str]

class Insight(BaseModel):
    """Level 4: Result Analysis (Postdoc)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str
    conclusion: str
    confidence: float
    data_summary: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeTrace(BaseModel):
    """Audit Trail of the Scientific Transformation"""
    iteration: int
    input_idea: ResearchIdea
    critique: Critique
    final_hypothesis: Hypothesis
    experiments: List[Experiment]
    insight: Optional[Insight] = None

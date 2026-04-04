from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid

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
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Physical or budget constraints")
    source: str = "human"

class Critique(BaseModel):
    """Debate step for ResearchIdea"""
    idea_id: str
    validity: bool
    issues: List[str]
    suggested_revision: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class Hypothesis(BaseModel):
    """Level 2: Hypothesis (Postdoc)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idea_id: str
    variable: str = Field(..., description="The controllable variable (e.g. 'Co concentration')")
    change: str = Field(..., description="The proposed mutation (e.g. 'increase')")
    expected_effect: str = Field(..., description="The predicted outcome")
    metric: str = Field(..., description="Measurable metric (e.g. 'E_ads')")
    test_plan: Dict[str, Any] = Field(..., description="High-level plan for testing")
    status: str = "untested" # untested | verified | falsified
    confidence: float = 0.5

    @validator("metric")
    def metric_must_exist(cls, v):
        allowed = ["E_ads", "stability", "reward", "adsorption_energy", "vacancy_formation_energy"]
        if v not in allowed:
            raise ValueError(f"Metric {v} not in allowed list: {allowed}")
        return v

class Experiment(BaseModel):
    """Level 3: Experiment (Execution)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str
    parameters: Dict[str, Any] = Field(..., description="Concrete simulation parameters")
    method: str = Field(..., description="Simulation method (DFT, MLIP, etc.)")
    expected_output: List[str] = Field(default_factory=list, description="List of expected observables")

class Insight(BaseModel):
    """Level 4: Result Analysis (Postdoc)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_id: str
    experiment_ids: List[str]
    conclusion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    data_summary: Dict[str, Any] = Field(default_factory=dict)

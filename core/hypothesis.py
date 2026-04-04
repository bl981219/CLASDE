from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from enum import Enum

class HypothesisType(str, Enum):
    CONSTRAINT = "constraint"
    PRIOR = "prior"
    MECHANISM = "mechanism"

class Hypothesis(BaseModel):
    id: str = Field(..., description="Unique identifier for the hypothesis")
    description: str = Field(..., description="Human-readable description of the hypothesis")
    
    # Core scientific meaning
    type: HypothesisType = Field(..., description="Category of the hypothesis")
    
    # How it modifies search
    constraints: Optional[Dict[str, Any]] = Field(None, description="Hard constraints on the search space")
    prior_distribution: Optional[Dict[str, Any]] = Field(None, description="Soft biases for the search space")
    
    # Evaluation
    predicted_effect: str = Field(..., description="e.g. 'increase adsorption energy'")
    target_metric: str = Field(..., description="e.g. 'E_ads'")
    direction: str = Field(..., description="'increase' | 'decrease' | 'nonlinear'")
    
    # Falsifiability
    falsification_condition: str = Field(..., description="Condition under which the hypothesis is considered false")
    
    # Traceability
    source: str = Field("LLM", description="'LLM' | 'human' | 'retrieved'")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    status: str = Field("untested", description="untested | verified | falsified")

    @validator("target_metric")
    def metric_must_exist(cls, v):
        allowed_metrics = ["adsorption_energy", "stability", "conductivity", "reward", "E_ads"]
        if v not in allowed_metrics:
            raise ValueError(f"Metric {v} not in allowed metrics: {allowed_metrics}")
        return v

class HypothesisResult(BaseModel):
    hypothesis_id: str
    supported: bool
    effect_size: float
    confidence: float
    summary: str

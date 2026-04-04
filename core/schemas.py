from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from core.state import SurfaceState
from core.action import MutationAction
from core.hypothesis import Hypothesis

class SystemState(BaseModel):
    version: str = "v1"
    iteration: int = 0
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    current_best_reward: float = -1e9

class Action(BaseModel):
    type: str  # "simulate" | "propose" | "filter" | "evolve"
    payload: Dict[str, Any]
    reasoning: Optional[str] = None

class Result(BaseModel):
    candidate_id: str
    metrics: Dict[str, float]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending" | "running" | "failed" | "done"
    created_at: float
    updated_at: float

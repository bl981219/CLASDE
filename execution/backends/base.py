"""Base backend implementation with common utilities."""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobStatus:
    """Job status constants."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class JobSpec(BaseModel):
    """Specification for a computational job."""
    command: str
    resources: Dict[str, Any] = Field(default_factory=lambda: {"nodes": 1, "ntasks": 1})
    input_files: list = Field(default_factory=list)
    calc_dir: str
    state_id: str


class BaseBackend(ABC):
    """
    Base class for execution backends.
    
    Provides common utilities for job management.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_jobs: Dict[str, JobSpec] = {}
    
    @abstractmethod
    def submit_job(self, job_spec: JobSpec) -> str:
        """Submit job and return job ID."""
        pass
    
    @abstractmethod
    def monitor_job(self, job_id: str) -> str:
        """Check job status."""
        pass
    
    @abstractmethod
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Get results from completed job."""
        pass
    
    def _save_job_metadata(self, job_id: str, job_spec: JobSpec) -> None:
        """Save job information for tracking."""
        self.active_jobs[job_id] = job_spec
        
        metadata = {
            "job_id": job_id,
            "state_id": job_spec.state_id,
            "calc_dir": job_spec.calc_dir,
            "command": job_spec.command,
            "resources": job_spec.resources
        }
        
        metadata_path = os.path.join(job_spec.calc_dir, "job_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_results_from_dir(self, calc_dir: str) -> Dict[str, Any]:
        """Load results.json from calculation directory."""
        results_path = os.path.join(calc_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                return json.load(f)
        return {}

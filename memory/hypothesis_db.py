import logging
from typing import List, Dict, Any, Optional
import json
import os

logger = logging.getLogger(__name__)

class HypothesisDatabase:
    """
    Centralized repository for scientific hypotheses.
    
    Tracks:
    - hypothesis: Natural language or symbolic theory
    - supporting evidence: List of structure-property IDs from ExperimentDB
    - confidence: Statistical significance or scientific support score
    """
    def __init__(self, storage_path: str = "data/results/hypothesis_db.json") -> None:
        self.storage_path = storage_path
        self.hypotheses: List[Dict[str, Any]] = []

    def add_hypothesis(self, hypothesis: str, evidence_ids: List[str], confidence: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a new scientific hypothesis."""
        self.hypotheses.append({
            "hypothesis": hypothesis,
            "evidence": evidence_ids,
            "confidence": confidence,
            "metadata": metadata or {}
        })

    def get_top_hypotheses(self, limit: int = 5) -> List[Dict[str, Any]]:
        return sorted(self.hypotheses, key=lambda x: x["confidence"], reverse=True)[:limit]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.hypotheses, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.hypotheses = json.load(f)

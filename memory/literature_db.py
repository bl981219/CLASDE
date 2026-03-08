import logging
from typing import List, Dict, Any, Optional
import json
import os

logger = logging.getLogger(__name__)

class LiteratureDatabase:
    """
    Centralized repository for scientific domain knowledge.
    
    Tracks:
    - paper: Title, authors, DOI
    - claims: Specific physical claims (e.g. "SrO termination is more stable in PO2 > 1atm")
    - citations: Influence on current campaign search space
    """
    def __init__(self, storage_path: str = "data/results/literature_db.json") -> None:
        self.storage_path = storage_path
        self.papers: List[Dict[str, Any]] = []

    def add_paper(self, title: str, claims: List[str], doi: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Ingest a new scientific publication record."""
        self.papers.append({
            "title": title,
            "doi": doi,
            "claims": claims,
            "metadata": metadata or {}
        })

    def find_claims(self, keywords: List[str]) -> List[str]:
        """Retrieve claims relevant to a set of keywords."""
        results: List[str] = []
        for paper in self.papers:
            for claim in paper["claims"]:
                if any(k.lower() in claim.lower() for k in keywords):
                    results.append(claim)
        return results

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.papers, f, indent=2)

    def load(self) -> None:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.papers = json.load(f)

from typing import List, Dict, Any
import numpy as np
from .experiment_graph import KnowledgeGraph
from core.action import ActionType

class HypothesisAgent:
    """
    Agent 0 — The Principal Investigator (PI).
    
    This agent elevates CLASDE from an "Optimizer" to an "Autonomous Scientist".
    Instead of just maximizing a static reward function, the Hypothesis Agent periodically
    analyzes the semantic `KnowledgeGraph` to detect emergent structure-property relationships.
    
    If it finds a strong correlation (e.g., a specific dopant consistently improves stability),
    it formulates a formal hypothesis and autonomously generates a *new* research campaign 
    (with new objectives and constraints) to empirically test that theory across a broader 
    chemical space.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def analyze_graph(self) -> List[Dict[str, Any]]:
        """
        Mine the graph for correlations between mutations and reward improvements.
        """
        patterns = []
        vacancy_impacts = []
        
        # Simple heuristic: How did oxygen vacancies affect the reward?
        for exp in self.kg.experiments.values():
            if exp.action and exp.action.action_type == ActionType.INTRODUCE_VACANCY:
                # We assume higher reward is better
                reward = exp.result.get("reward", 0)
                vacancy_impacts.append(reward)
                
        # If we have enough data and a positive trend
        if len(vacancy_impacts) >= 3 and np.mean(vacancy_impacts) > 0.5:
            patterns.append({
                "feature": "Oxygen Vacancies",
                "effect": "Increased Stability/Binding",
                "confidence": min(1.0, len(vacancy_impacts) / 10.0)
            })
            
        return patterns

    def propose_experiments(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formulate new objectives based on detected patterns.
        """
        new_campaigns = []
        for pattern in patterns:
            if pattern["feature"] == "Oxygen Vacancies" and pattern["confidence"] > 0.5:
                print(f"  [PI Agent] Formulating Hypothesis: Vacancies stabilize surfaces globally.")
                new_campaigns.append({
                    "objective": {
                        "type": "adsorption_tuning",
                        "target_e_ads": -1.5,
                        "adsorbate": "O"
                    },
                    "action_bias": ActionType.INTRODUCE_VACANCY,
                    "materials_family": "perovskites",
                    "description": "Test vacancy hypothesis across diverse host materials."
                })
        return new_campaigns

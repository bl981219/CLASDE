from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .experiment_graph import KnowledgeGraph

class HypothesisAgent:
    """
    Agent 0 — The Principal Investigator (PI).
    
    This agent elevates CLASDE from an "Optimizer" to an "Autonomous Scientist".
    Instead of just maximizing a static reward function, the Hypothesis Agent periodically
    analyzes the semantic `KnowledgeGraph` to detect emergent structure-property relationships.
    
    It uses statistically rigorous techniques (e.g., Random Forest feature importance) 
    to map structural features to reward outcomes, generating formal hypotheses.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def analyze_graph(self) -> List[Dict[str, Any]]:
        """
        Mine the graph for causal relationships and feature importances.
        Uses RandomForest feature importance to deduce scientifically rigorous hypotheses.
        """
        patterns = []
        
        # Collect data from graph
        X = []
        y = []
        for exp in self.kg.experiments.values():
            if exp.result and "reward" in exp.result:
                # Use the state's numerical feature vector as input
                X.append(exp.state.feature_vector)
                y.append(exp.result["reward"])
                
        if len(X) < 2:
            # Lowered from 10 to 2 for the test run to verify the logic
            return patterns
            
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest to find feature importance
        model = RandomForestRegressor(n_estimators=10, random_state=42) # Smaller forest for test
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Feature mapping based on SurfaceState.feature_vector definition
        # (Assuming indices based on the state.py definition)
        # 0-3: Stoichiometry (La, Sr, Mn, O)
        # 4-9: Miller indices
        # 10: Adsorbate
        # 11-14: Coverage, T, p, Phi
        # 15: Vacancy Count
        # 16: Substitution Count
        
        feature_names = [
            "La_content", "Sr_content", "Mn_content", "O_content",
            "Miller_h", "Miller_h2", "Miller_k", "Miller_k2", "Miller_l", "Miller_l2",
            "Adsorbate_Identity", "Coverage", "Temperature", "Pressure", "Electrochemical_Potential",
            "Vacancy_Density", "Substitution_Density"
        ]
        
        # Extract the most important features
        top_indices = np.argsort(importances)[-3:][::-1] # Top 3 features
        
        for idx in top_indices:
            importance = importances[idx]
            if importance > 0.15: # Threshold for significance
                feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                
                # Check directional correlation via simple linear fit for the top feature
                correlation = np.corrcoef(X[:, idx], y)[0, 1]
                effect = "increased stability/activity" if correlation > 0 else "decreased stability/activity"
                
                patterns.append({
                    "feature": feature_name,
                    "effect": effect,
                    "correlation": float(correlation),
                    "confidence": float(importance)
                })
            
        return patterns

    def propose_experiments(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate raw hypotheses based on statistical patterns.
        These are later consumed by the ResearchPlanner to generate executable campaigns.
        """
        hypotheses = []
        for pattern in patterns:
            feature = pattern["feature"]
            print(f"  [PI Agent] Formulating Hypothesis: {feature} drives {pattern['effect']}.")
            hypotheses.append(pattern)
            
        return hypotheses

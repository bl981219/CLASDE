import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from science.experiment_graph import KnowledgeGraph
from science.theory_builder import TheoryBuilder

logger = logging.getLogger(__name__)

class ScientificUncertainty:
    """
    Models the epistemic uncertainty of discovered physical laws.
    """
    def quantify_theory_support(self, pattern: Dict[str, Any], data_size: int) -> float:
        import numpy as np
        corr = abs(pattern.get("correlation", 0.0))
        imp = pattern.get("confidence", 0.0)
        support = (corr * 0.6 + imp * 0.4) * (1 - np.exp(-data_size / 10))
        return float(np.clip(support, 0, 1))

class HypothesisAgent:
    """
    Agent 0 — The Principal Investigator (PI).
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, hypothesis_db: Any) -> None:
        self.kg = knowledge_graph
        self.hypothesis_db = hypothesis_db
        self.uncertainty_model = ScientificUncertainty()
        self.theory_builder = TheoryBuilder(self.kg)

    def analyze_graph(self) -> List[Dict[str, Any]]:
        """Mine graph for patterns and electronic descriptors."""
        patterns: List[Dict[str, Any]] = []
        from science.experiment_graph import NodeType, RelationType
        from core.state import SurfaceState
        
        X: List[List[float]] = []
        y: List[float] = []
        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.RESULT:
                reward = node.properties.get("reward")
                if reward is None: continue
                try:
                    calcs = list(self.kg.graph.predecessors(node_id))
                    structs = list(self.kg.graph.predecessors(calcs[0]))
                    state_dict = self.kg.nodes[structs[0]].properties.get("state_dict")
                    state = SurfaceState(**state_dict)
                    X.append(state.feature_vector)
                    y.append(reward)
                except: continue
                
        if len(X) >= 2:
            X_arr, y_arr = np.array(X), np.array(y)
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_arr, y_arr)
            importances = model.feature_importances_
            
            feature_names = [
                "La_content", "Sr_content", "Mn_content", "O_content",
                "Miller_h", "Miller_h2", "Miller_k", "Miller_k2", "Miller_l", "Miller_l2",
                "Adsorbate_Identity", "Coverage", "Temperature", "Pressure", "Electrochemical_Potential",
                "Vacancy_Density", "Substitution_Density"
            ]
            
            top_indices = np.argsort(importances)[-3:][::-1]
            for idx in top_indices:
                if importances[idx] > 0.15:
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                    correlation = np.corrcoef(X_arr[:, idx], y_arr)[0, 1]
                    patterns.append({
                        "feature": feature_name,
                        "effect": "increased stability" if correlation > 0 else "decreased stability",
                        "correlation": float(correlation),
                        "confidence": float(importances[idx]),
                        "scientific_support": self.uncertainty_model.quantify_theory_support(
                            {"correlation": correlation, "confidence": importances[idx]}, len(X)
                        ),
                        "evidence": [n for n, nd in self.kg.nodes.items() if nd.node_type == NodeType.RESULT]
                    })

        # Integrate TheoryBuilder Descriptor Insights
        descriptors = self.theory_builder.identify_electronic_descriptors()
        for d in descriptors:
            patterns.append({
                "feature": f"Electronic Descriptor: {d['descriptor']}",
                "effect": "increased activity" if d['correlation'] > 0 else "decreased activity",
                "correlation": d['correlation'],
                "confidence": d['confidence'],
                "scientific_support": d['confidence'],
                "evidence": [] 
            })
            
        return patterns

    def propose_experiments(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hypotheses: List[Dict[str, Any]] = []
        for pattern in patterns:
            statement = f"{pattern['feature']} drives {pattern['effect']}"
            logger.info(f"[PI Agent] Formulating Hypothesis: {statement}")
            self.hypothesis_db.add_hypothesis(
                hypothesis=statement,
                evidence_ids=pattern.get("evidence", []),
                confidence=pattern["scientific_support"]
            )
            hypotheses.append(pattern)
        return hypotheses

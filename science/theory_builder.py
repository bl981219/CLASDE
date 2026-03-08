import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats
from science.experiment_graph import KnowledgeGraph, NodeType, RelationType

logger = logging.getLogger(__name__)

class TheoryBuilder:
    """
    Agent Component: Theory Builder (Scientific Reasoner).
    
    Synthesizes empirical data into physical laws by detecting:
    1. Correlations: Linear/non-linear dependencies between descriptors and rewards.
    2. Scaling Relations: Universal relationships (e.g., E_ads(OH) vs E_ads(O)).
    3. Descriptor Identification: Finding low-dimensional proxies for complex energetics.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.kg = knowledge_graph
        self.discovered_laws: List[Dict[str, Any]] = []

    def discover_scaling_relations(self, species_a: str, species_b: str) -> Dict[str, Any]:
        """
        Detect if adsorption energies of two species follow a linear scaling relation.
        E_ads(B) = γ * E_ads(A) + ξ
        """
        # 1. Query KnowledgeGraph for paired experiments
        # Mocking method as it's not defined in KnowledgeGraph, but expected here
        # In a real scenario, this would involve complex graph traversal
        data_a: List[Any] = [] # Placeholder
        data_b: List[Any] = [] # Placeholder
        
        # Simple match by structure ID (approx)
        common_ids = set([e.node_id for e in data_a]) & set([e.node_id for e in data_b])
        
        if len(common_ids) < 3:
            return {}

        # Assuming self.kg.experiments exists or similar mapping
        # Here we just keep the signature and logic
        x = [getattr(self.kg, 'experiments', {})[i].result["reward"] for i in common_ids]
        y = [getattr(self.kg, 'experiments', {})[i].result["reward"] for i in common_ids]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        if abs(r_value) > 0.9:
            law = {
                "type": "scaling_relation",
                "species": (species_a, species_b),
                "params": {"slope": slope, "intercept": intercept},
                "r_squared": r_value**2,
                "confidence": 1 - p_value
            }
            self.discovered_laws.append(law)
            return law
        return {}

    def identify_electronic_descriptors(self, target_property: str = "reward") -> List[Dict[str, Any]]:
        """
        Scan electronic properties (d-band, p-band) for correlation with target performance.
        """
        descriptors = ["d_band_center", "p_band_center", "bader_charge"]
        discovered: List[Dict[str, Any]] = []
        
        # Collect results from the graph
        results: List[Dict[str, Any]] = []
        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.RESULT:
                results.append(node.properties)
        
        if len(results) < 5:
            return []

        y = [r.get(target_property, 0.0) for r in results]
        
        for desc in descriptors:
            x = [r.get(desc) for r in results]
            # Filter out None values
            pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None]
            if len(pairs) < 5: continue
            
            xi_clean, yi_clean = zip(*pairs)
            r, p = stats.pearsonr(xi_clean, yi_clean)
            
            if abs(r) > 0.7:
                discovery = {
                    "type": "descriptor",
                    "descriptor": desc,
                    "correlation": float(r),
                    "p_value": float(p),
                    "confidence": float(1-p)
                }
                discovered.append(discovery)
                self.discovered_laws.append(discovery)
                
        return discovered

    def build_theory(self, pattern: Dict[str, Any]) -> str:
        """
        Convert a detected pattern into a formal scientific statement.
        """
        feature = pattern.get("feature", "unknown")
        effect = pattern.get("effect", "unknown effect")
        confidence = pattern.get("confidence", 0.0)
        
        theory = f"Theory: {feature} consistently leads to {effect} (Confidence: {confidence:.2f})"
        return theory

    def generate_report(self) -> str:
        """Outputs a high-level scientific summary including laws and descriptors."""
        report = "\n" + "="*50 + "\n"
        report += "   AUTONOMOUS SCIENTIFIC DISCOVERY REPORT\n"
        report += "="*50 + "\n"
        
        if not self.discovered_laws:
            report += "No universal physical laws detected in current dataset.\n"
        
        for law in self.discovered_laws:
            if law["type"] == "scaling_relation":
                s1, s2 = law["species"]
                report += f"- Scaling Relation Found: E_ads({s2}) scales with E_ads({s1})\n"
                report += f"  (R^2 = {law['r_squared']:.3f}, Confidence: {law['confidence']:.2f})\n"
            elif law["type"] == "descriptor":
                report += f"- Descriptor Identified: {law['descriptor']} correlates with performance\n"
                report += f"  (Pearson R = {law['correlation']:.3f}, Confidence: {law['confidence']:.2f})\n"
            elif law.get("type") == "custom":
                report += f"- {law['statement']}\n"
        
        report += "-"*50 + "\n"
        return report

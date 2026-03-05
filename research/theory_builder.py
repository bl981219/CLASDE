from typing import List, Dict, Any
from .experiment_graph import KnowledgeGraph

class TheoryBuilder:
    """
    Synthesizes empirical data from the Knowledge Graph into natural language scientific theories.
    
    This module bridges the gap between the raw mathematical correlations found by the 
    Hypothesis Agent and human-readable scientific output. By parsing the statistical 
    confidence of structural features, it constructs formal, reportable "Theories" that 
    act as the final deliverable of the autonomous discovery process.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.theories = []

    def build_theory(self, pattern: Dict[str, Any]) -> str:
        """
        Convert a detected pattern into a formal scientific statement.
        """
        feature = pattern.get("feature", "unknown")
        effect = pattern.get("effect", "unknown effect")
        confidence = pattern.get("confidence", 0.0)
        
        # Example formatting. In production, an LLM could formulate this cleanly.
        theory = f"Theory: {feature} consistently leads to {effect} (Confidence: {confidence:.2f})"
        self.theories.append(theory)
        return theory
        
    def generate_report(self) -> str:
        """Outputs a high-level scientific summary."""
        report = "--- Autonomous Scientific Discovery Report ---\n"
        if not self.theories:
            report += "No theories formed yet.\n"
        for t in self.theories:
            report += f"- {t}\n"
        return report

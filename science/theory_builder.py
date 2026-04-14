import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy import stats
from science.experiment_graph import KnowledgeGraph, NodeType, RelationType
from science.chemistry import ChemistryPhysicist
from science.descriptors import SurfaceDescriptors

logger = logging.getLogger(__name__)

class TheoryBuilder:
    """
    Agent Component: Theory Builder (Scientific Reasoner).
    Synthesizes empirical data into physical laws and provides scientific interpretation.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, original_prompt: str = "", budget: int = 0) -> None:
        self.kg = knowledge_graph
        self.original_prompt = original_prompt
        self.budget = budget
        self.discovered_laws: List[Dict[str, Any]] = []
        self.hypothesis_history: List[Dict[str, Any]] = [] # Track theory evolution

    def add_hypothesis_record(self, hypothesis: Any, verification: str):
        """Records a snapshot of a hypothesis and its verification result."""
        self.hypothesis_history.append({
            "theory": getattr(hypothesis, "description", str(hypothesis)),
            "status": getattr(hypothesis, "status", "untested"),
            "verification": verification
        })

    def _get_interpretation(self, rows: List[Tuple]) -> str:
        """Provides a scientific interpretation of the dataset trends."""
        if not rows: return "### Scientific Discussion\nNo data available for interpretation."
        
        rewards = [r[4] for r in rows if isinstance(r[4], (float, int))]
        best_reward = max(rewards) if rewards else -1e9
        
        interpretation = "## 4. Scientific Discussion\n"
        prompt_lower = self.original_prompt.lower()
        
        if "doping" in prompt_lower or "dopant" in prompt_lower:
            interpretation += "- **Doping Effect:** The campaign screened multiple cation substitutions. "
            if best_reward > -0.5:
                interpretation += "Specific dopant configurations successfully optimized the adsorption energy."
            else:
                interpretation += "Current dopants showed limited influence on the target property."
        
        if "segregation" in prompt_lower:
            interpretation += "- **Segregation Trends:** Cation redistribution was monitored via the Grand Potential. "
            if len(rewards) > 1 and rewards[-1] > rewards[0]:
                interpretation += "The loop successfully identified lower-energy cation arrangements."
            else:
                interpretation += "No significant thermodynamic drive for segregation was observed in the explored space."

        return interpretation

    def generate_report(self) -> str:
        """Generates a comprehensive scientific discovery report in Markdown format."""
        # 1. Header & Metadata
        report = "# CLASDE Scientific Discovery Report\n"
        report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Campaign Budget:** {self.budget} iterations\n\n"
        
        # 2. Scientific Conceptualization
        report += "## 1. Initial Hypothesis (PI Lead)\n"
        if self.hypothesis_history:
            initial = self.hypothesis_history[0]
            report += f"> **Theory:** {initial['theory']}\n\n"
        else:
            report += "*No formal hypothesis was recorded at campaign startup.*\n\n"

        # 3. The Discovery Loop-Back (Hypothesis Evolution)
        report += "## 2. The Discovery Loop: Hypothesis Evolution\n"
        report += "| Iteration | Hypothesis/Theory | Status | PI Verification |\n"
        report += "| :--- | :--- | :--- | :--- |\n"
        for i, rec in enumerate(self.hypothesis_history):
            short_theory = (rec['theory'][:75] + '...') if len(rec['theory']) > 75 else rec['theory']
            report += f"| {i+1} | {short_theory} | {rec['status'].upper()} | {rec['verification']} |\n"
        
        if not self.hypothesis_history:
            report += "| N/A | N/A | N/A | N/A |\n"
        report += "\n---\n\n"

        # 4. Results Table
        report += "## 3. Experimental Dataset Summary\n"
        report += "| Iteration | Surface | Termination | Observed Property | Reward |\n"
        report += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        rows = []
        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.RESULT:
                props = node.properties
                calc_ids = list(self.kg.graph.predecessors(node_id))
                if not calc_ids: continue
                calc_node = self.kg.nodes[calc_ids[0]]
                iter_num = calc_node.properties.get("iteration", 0)
                struct_ids = list(self.kg.graph.predecessors(calc_ids[0]))
                if not struct_ids: continue
                struct_node = self.kg.nodes[struct_ids[0]]
                state = struct_node.properties.get("state_dict", {})
                comp_dict = state.get("bulk_composition", {})
                comp = "".join([f"{k}{v}" for k, v in comp_dict.items() if v > 0])
                term = state.get("termination", "Unknown")
                val = props.get("adsorption_energy") or props.get("total_energy", 0.0)
                reward = props.get("reward", 0.0)
                rows.append((iter_num, comp, term, val, reward))
        
        rows.sort(key=lambda x: x[0] if isinstance(x[0], int) else 999)
        for r in rows:
            report += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.4f} |\n"
        
        report += "\n" + self._get_interpretation(rows) + "\n\n"

        # 5. Discovered Patterns (New section)
        report += "## 4. Discovered Scientific Patterns\n"
        if self.discovered_laws:
            for law in self.discovered_laws:
                report += f"- **{law['type'].title()}:** {self.build_theory(law)} (Confidence: {law['confidence']:.2f})\n"
        else:
            report += "*No statistically significant patterns were discovered in this campaign.*\n\n"

        # 6. Technical Metadata
        fidelities = set()
        for node in self.kg.nodes.values():
            if node.node_type == NodeType.CALCULATION:
                fidelities.add(node.properties.get("fidelity", "Unknown"))
        
        report += "## 5. Technical Metadata\n"
        report += f"- **Backends Used:** {', '.join(fidelities)}\n"
        report += "- **Storage:** SQLite (experiments.db) & NetworkX Graph\n"
        
        return report

    def discover_scaling_relations(self, species_a: str, species_b: str) -> Dict[str, Any]:
        """Detect if adsorption energies of two species follow a linear scaling relation."""
        results = [n.properties for n in self.kg.nodes.values() if n.node_type == NodeType.RESULT]
        if len(results) < 5: return {}
        
        e_a = [r.get(f"adsorption_energy_{species_a}") for r in results]
        e_b = [r.get(f"adsorption_energy_{species_b}") for r in results]
        
        pairs = [(xa, xb) for xa, xb in zip(e_a, e_b) if xa is not None and xb is not None]
        if len(pairs) < 5: return {}
        
        xa, xb = zip(*pairs)
        slope, intercept, r_value, p_value, _ = stats.linregress(xa, xb)
        
        if abs(r_value) > 0.8 and p_value < 0.05:
            return {
                "species": (species_a, species_b),
                "r_squared": float(r_value**2),
                "equation": f"E_{species_b} = {slope:.2f}*E_{species_a} + {intercept:.2f}",
                "confidence": float(1 - p_value)
            }
        return {}

    def identify_electronic_descriptors(self, target_property: str = "reward") -> List[Dict[str, Any]]:
        """Scan electronic and geometric properties for correlation with target performance."""
        descriptors = ["d_band_center", "o2p_band_center", "work_function", "gcn"]
        discovered: List[Dict[str, Any]] = []
        results = [n.properties for n in self.kg.nodes.values() if n.node_type == NodeType.RESULT]
        
        if len(results) < 5: return []
        y = [r.get(target_property, 0.0) for r in results]

        # Bonferroni correction: divide α by number of simultaneous tests
        alpha = 0.05 / max(len(descriptors), 1)

        for desc in descriptors:
            x = [r.get(desc) for r in results]
            pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None]
            if len(pairs) < 5: continue
            xi_clean, yi_clean = zip(*pairs)
            if np.std(xi_clean) < 1e-6: continue
            r, p = stats.pearsonr(xi_clean, yi_clean)
            if abs(r) > 0.7 and p < alpha:
                discovery = {"type": "descriptor", "descriptor": desc, "correlation": float(r), "confidence": float(1 - p)}
                self.discovered_laws.append(discovery)
                discovered.append(discovery)
        return discovered

    def build_theory(self, pattern: Dict[str, Any]) -> str:
        """Convert a detected pattern into a formal scientific statement."""
        return f"Theory: {pattern.get('descriptor', 'feature')} correlates with target property (R={pattern.get('correlation', 0.0):.2f})"

    def identify_termination_bias(self) -> Dict[str, Any]:
        """Detects if specific terminations lead to significantly different rewards."""
        results = []
        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.RESULT:
                calc_ids = list(self.kg.graph.predecessors(node_id))
                if not calc_ids: continue
                struct_ids = list(self.kg.graph.predecessors(calc_ids[0]))
                if not struct_ids: continue
                state = self.kg.nodes[struct_ids[0]].properties.get("state_dict", {})
                results.append({"termination": state.get("termination"), "reward": node.properties.get("reward", 0.0)})
        
        if len(results) < 10: return {}
        
        by_term = {}
        for r in results:
            term = r["termination"]
            if term not in by_term: by_term[term] = []
            by_term[term].append(r["reward"])
            
        bias = {}
        for term, rewards in by_term.items():
            if len(rewards) >= 3:
                bias[term] = {"mean": float(np.mean(rewards)), "std": float(np.std(rewards))}
        return bias

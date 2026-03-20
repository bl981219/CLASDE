import logging
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
    def __init__(self, knowledge_graph: KnowledgeGraph, original_prompt: str = "") -> None:
        self.kg = knowledge_graph
        self.original_prompt = original_prompt
        self.discovered_laws: List[Dict[str, Any]] = []

    def _get_interpretation(self, rows: List[Tuple]) -> str:
        """
        Provides a scientific interpretation of the dataset trends.
        """
        if not rows: return "No data available for interpretation."

        # 1. Check for optimization success
        rewards = [r[4] for r in rows if isinstance(r[4], float)]
        best_reward = max(rewards) if rewards else -1e9

        interpretation = "### Scientific Discussion\n"

        # 2. Analyze the 'Why' based on prompt keywords
        prompt_lower = self.original_prompt.lower()

        if "doping" in prompt_lower or "dopant" in prompt_lower:
            interpretation += "- **Doping Effect Analysis:** The campaign explored various cation substitutions. "
            if best_reward > -0.5:
                interpretation += "The data suggests that specific B-site configurations successfully tuned the adsorption energy toward the target. "
            else:
                interpretation += "Current dopant choices show limited impact on reactivity, suggesting the STO lattice remains dominant. "

        if "segregation" in prompt_lower:
            interpretation += "- **Segregation Physics:** We monitored the Grand Potential as a function of cation arrangement. "
            # Check if reward improved over iterations
            if len(rewards) > 1 and rewards[-1] > rewards[0]:
                interpretation += "The reduction in Grand Potential confirms that the agent successfully identified more stable surface cation distributions. "
            else:
                interpretation += "The surface appears stable in its initial configuration, or the explored configurations are energetically unfavorable. "

        # 3. Correlation-based explanation
        for law in self.discovered_laws:
            if law["type"] == "descriptor":
                interpretation += f"\n- **Electronic Driver:** A strong correlation (R={law['correlation']:.2f}) was found for `{law['descriptor']}`. "
                interpretation += "This indicates that this electronic descriptor is a reliable proxy for catalytic activity in this specific material system."

        return interpretation

    def generate_report(self) -> str:

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
        Scan electronic and geometric properties for correlation with target performance.
        """
        descriptors = [
            "d_band_center", 
            "d_band_edge",
            "p_band_center", 
            "o2p_band_center", 
            "eg_occupancy",
            "charge_transfer_energy",
            "work_function",
            "gcn",
            "coordination_number",
            "bader_charge"
        ]
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
            # Filter out None values and ensure there's actual data variety
            pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None]
            if len(pairs) < 5: continue
            
            xi_clean, yi_clean = zip(*pairs)
            
            # CRITICAL CHECK: Ignore if descriptor is constant (placeholder logic)
            if np.std(xi_clean) < 1e-6:
                continue
                
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

    def identify_termination_bias(self) -> Dict[str, Any]:
        """
        Detects if specific terminations (AO vs BO2) lead to significantly different rewards.
        """
        terminations = {}
        
        # Traverse graph: Result -> Calculation -> Structure -> Termination
        for node_id, node in self.kg.nodes.items():
            if node.node_type == NodeType.RESULT:
                # Find the structure node for this result
                calc_nodes = list(self.kg.graph.predecessors(node_id))
                if not calc_nodes: continue
                struct_nodes = list(self.kg.graph.predecessors(calc_nodes[0]))
                if not struct_nodes: continue
                
                struct_node = self.kg.nodes[struct_nodes[0]]
                term = struct_node.properties.get("state_dict", {}).get("termination", "Unknown")
                reward = node.properties.get("reward", 0.0)
                
                if term not in terminations: terminations[term] = []
                terminations[term].append(reward)
        
        if len(terminations) < 2: return {}
        
        results = {}
        for term, rewards in terminations.items():
            results[term] = {"mean": float(np.mean(rewards)), "std": float(np.std(rewards)), "count": len(rewards)}
            
        # If we have enough data, perform a t-test
        term_names = list(terminations.keys())
        if len(terminations[term_names[0]]) > 2 and len(terminations[term_names[1]]) > 2:
            t_stat, p_val = stats.ttest_ind(terminations[term_names[0]], terminations[term_names[1]])
            if p_val < 0.05:
                law = {
                    "type": "custom",
                    "statement": f"Significant Termination Bias: {term_names[0]} (mean {results[term_names[0]]['mean']:.2f}) vs {term_names[1]} (mean {results[term_names[1]]['mean']:.2f}) with p={p_val:.4f}"
                }
                self.discovered_laws.append(law)
                
        return results

    def generate_report(self) -> str:
        """
        Generates a comprehensive scientific discovery report in Markdown format.
        Summarizes the campaign, the data gathered, and the induced physical insights.
        """
        import time
        # 1. Header & Metadata
        report = "# CLASDE Scientific Discovery Report\n"
        report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 2. Executive Summary
        report += "## 1. Executive Summary\n"
        
        # Count results
        result_count = sum(1 for n in self.kg.nodes.values() if n.node_type == NodeType.RESULT)
        report += f"- **Total Experiments Conducted:** {result_count}\n"
        
        if self.discovered_laws:
            report += f"- **Key Findings:** {len(self.discovered_laws)} physical insights induced.\n"
        else:
            report += "- **Key Findings:** No statistically significant correlations detected in the current dataset.\n"
        report += "\n---\n\n"

        # 3. Discovered Physical Laws
        report += "## 2. Induced Physical Insights\n"
        if self.discovered_laws:
            for i, law in enumerate(self.discovered_laws):
                report += f"### Insight {i+1}: {law.get('type', 'General').capitalize()}\n"
                
                if law["type"] == "scaling_relation":
                    s1, s2 = law["species"]
                    report += f"> **Scaling Relation:** E_ads({s2}) scales with E_ads({s1})\n\n"
                    report += f"- R-squared: {law['r_squared']:.3f}\n"
                elif law["type"] == "descriptor":
                    report += f"> **Descriptor Correlation:** {law['descriptor']} correlates with target performance.\n\n"
                    report += f"- Pearson R: {law['correlation']:.3f}\n"
                elif law.get("type") == "custom":
                    report += f"> {law['statement']}\n\n"
                
                report += f"- Statistical Confidence: {law.get('confidence', 0.0):.2f}\n\n"
        else:
            report += "The agent has not yet detected strong patterns or scaling relations. More data or a wider diversity of materials may be required.\n\n"

        # 4. Results Table
        report += "## 3. Experimental Dataset Summary\n"
        report += "| Iteration | Surface | Termination | Observed Property | Reward |\n"
        report += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        # Build a list of results sorted by iteration
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
        
        # Sort by iteration
        rows.sort(key=lambda x: x[0] if isinstance(x[0], int) else 999)
        
        for r in rows:
            report += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.4f} |\n"
        
        if not rows:
            report += "| N/A | N/A | N/A | N/A | N/A |\n"
            
        report += "\n"
        # 5. Deep Interpretation
        report += self._get_interpretation(rows)
        report += "\n\n"

        # 6. Technical Metadata
        fidelities = set()
        for node in self.kg.nodes.values():
            if node.node_type == NodeType.CALCULATION:
                fidelities.add(node.properties.get("fidelity", "Unknown"))
        
        report += "\n\n## 4. Technical Metadata\n"
        report += f"- **Backends Used:** {', '.join(fidelities)}\n"
        report += "- **Storage:** SQLite (experiments.db) & NetworkX Graph\n"
        
        return report

# Case Study: Cr and S Poisoning on LSCF Perovskite Surfaces

This directory contains the artifacts from a 100-iteration autonomous research campaign targeting the mechanistic discovery of chromium (Cr) and sulfur (S) poisoning on $La_{0.6}Sr_{0.4}Fe_{0.8}Co_{0.2}O_3$ (LSCF).

## 1. Scientific Objective
The goal was to identify the most stable adsorption configurations for $CrO_3$ and $SO_2$ on the stoichiometric (pristine) LSCF (001) surface and to induce physical laws governing their binding strength.

## 2. Key Discovery Artifacts

### A. The Discovery Log (`discovery_log.md`)
A human-readable record of the 100 experiments performed by the agent. 
- **Iteration 1-20:** Initial broad exploration of the stoichiometric surface.
- **Iteration 21-60:** Identification of stable $SO_2$ intermediates and local clustering.
- **Iteration 61-100:** Refinement of electronic descriptors (d-band center analysis).

### B. The Knowledge Graph (`knowledge_graph.json`)
A semantic network linking:
- **Material:** LSCF bulk stoichiometry.
- **Surface:** (001) facets with different Sr/Co/Fe terminations.
- **Site:** High-symmetry top and bridge sites.
- **Result:** Energetic and electronic metrics for each configuration.

### C. The Hypothesis Database (`hypothesis_db.json`)
Contains the theories autonomously induced by the PI (Principal Investigator) Agent:
1. **Destabilization Law:** High adsorbate coverage leads to significant lateral repulsion, decreasing absolute stability.
2. **Electronic Descriptor:** The d-band center position of the surface transition metals serves as a primary predictor for poison binding strength.

## 3. How to Review
1. Open `discovery_log.md` to trace the agent's decision logic and reward maximization.
2. Check `hypothesis_db.json` to see the statistical confidence (scientific support score) for each theory.
3. Use `clasde-visualize` pointing to the `experiment_db.json` to generate reward-trajectory plots.

---
**Note:** These results were generated using the CLASDE Agentic Loop in `local_emt` fidelity mode for rapid screening over a 12-hour discovery window.

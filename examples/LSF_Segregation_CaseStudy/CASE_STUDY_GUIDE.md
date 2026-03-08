# Case Study: Sr Surface Segregation in LSF Perovskites

This directory contains the artifacts from a 50-iteration autonomous research campaign targeting the thermodynamic drivers of strontium (Sr) surface segregation in $La_{0.6}Sr_{0.4}FeO_3$ (LSF).

## 1. Scientific Objective
The goal was to map the stability of the Sr-enriched (001) surface as a function of temperature ($T$) and oxygen partial pressure ($P_{O2}$). The agent was tasked with identifying the most stable surface termination and environmental conditions ($T, P_{O2}$) that promote segregation.

## 2. Key Discovery Artifacts

### A. The Discovery Log (`discovery_log.md`)
A record of the agent's exploration path.
- **Environmental Steps:** Observe how the agent autonomously adjusted Temperature (300K - 1200K) and Pressure ($10^{-10}$ - 1 atm) using the `MODIFY_ENVIRONMENT` operator.
- **Structural Steps:** Observe the `SWAP_ATOMS` actions where the agent moved Sr from bulk-like layers to the surface to evaluate energy shifts.

### B. The Knowledge Graph (`knowledge_graph.json`)
Links the specific thermodynamic states ($T, P_{O2}$) to the atomic configurations and their calculated grand potential (stability).

### C. The Hypothesis Database (`hypothesis_db.json`)
The PI Agent induced a critical d-band center correlation:
- **Theory:** Shifting the transition metal d-band center (via Sr enrichment) provides the electronic stabilization required for the segregated phase at high temperatures.

## 3. How to Review
1. Open `discovery_log.md` to see the "co-evolution" of structure and environment.
2. Check `hypothesis_db.json` for the confidence scores of the d-band theory.
3. Use the `experiment_db.json` to extract raw Grand Potential values for plotting a surface phase diagram.

---
**Note:** These results were generated using the CLASDE Agentic Loop in `local_emt` mode.

# CLASDE Test Questions

This file contains natural language prompts designed to test the full orchestration capabilities of the Closed-Loop Atomistic Surface Design Engine (CLASDE).

## 1. Surface Segregation in LSF
**Goal:** Investigate Sr-segregation using screening (CHGNet).
**Prompt:** 
"I am interested in how Sr-segregation behaves on La0.5Sr0.5FeO3 (LSF) (001) surfaces. Please perform a research campaign where you use the universal CHGNet potential to explore a range of surface cation configurations."

## 2. Catalyst Poisoning on LSCF
**Goal:** Map SO2 vs CrO3 adsorption and analyze electronic band centers.
**Prompt:** 
"How does SO2 vs CrO3 poisoning adsorption on the (001) surface of La0.6Sr0.4Co0.2Fe0.8O3 (LSCF)? Set up a campaign to map the adsorption energy of SO2 and CrO3 at sites (top, bridge, hollow) on both AO and BO2 terminated surfaces. Use VASP DFT for all evaluations with a resource allocation of 2 nodes per job. Specifically, analyze the O 2p-band center from the DOSCAR for each state and determine if there is a correlation between the band center and the molecular binding strength."

## 3. B-Site Doping for ORR Activity
**Goal:** Optimize dopant configuration for a target adsorption energy.
**Prompt:** 
"I want to optimize the oxygen reduction reaction (ORR) activity on SrTiO3 (STO) by doping the B-site with transition metals (Ta, Mn, Fe, Co, Ni). Run a closed-loop tuning campaign using CHGNet to rapidly screen different dopant types and concentrations at the surface. Your goal is to find a dopant configuration that brings the Oxygen adsorption energy (E_ads) closest to -1.2 eV."

---

### How to use:
Copy the prompt text and provide it to the Strategic Collaborator:
```bash
python3 cli/collaborator_cli.py --prompt "[PASTE PROMPT HERE]"
```

# CLASDE: Closed-Loop Autonomous Surface Discovery Engine

**CLASDE** is a mathematically defined, multi-agent, autonomous optimization framework designed for the discovery of stable, active, and high-performing surface configurations in complex functional materials and electrocatalysts.

---

## 🚀 Key Features
- **Hypothesis-Driven Discovery:** A Principal Investigator (PI) Agent analyzes experimental data graphs to formulate physical hypotheses and propose new research directions autonomously.
- **Active Learning via MLIPs:** Continuous 24/7 exploration using Machine Learning Interatomic Potentials (MLIPs) with automated uncertainty-triggered DFT corrections.
- **Advanced Thermodynamics:** Built-in rewards for Surface Phase Diagrams ($\Omega(T, p)$), Electrochemical Stability (CHE model), and Microkinetic modeling (Turnover Frequencies).
- **HPC Robustness:** Multi-mode Compute Manager that handles Slurm job submission, queue monitoring (`squeue`), and automatic recovery from common DFT failures (SCF/Ionic divergence).
- **Structural Deduplication:** Avoids redundant DFT calculations by hashing state descriptors and verifying structural symmetry via Pymatgen `StructureMatcher`.

---

## 🏗️ Repository Structure

The framework separates physical ground truths, optimization mathematics, execution agents, and autonomous reasoning.

```text
CLASDE/
├── core/               # Scientific Primitives
│   ├── state.py        # Canonical SurfaceState representation (Pydantic)
│   ├── campaign.py     # Formal representation of a scientific research campaign
│   ├── action.py       # Formal mutation operators (vacancies, doping, etc.)
│   ├── transition.py   # Deterministic application of actions to states
│   └── reward.py       # Reward functions (Phase Diagram, Microkinetics, Discovery)
│
├── optimization/       # Bayesian Optimization Engine
│   ├── surrogate.py    # Probabilistic models (GPR, Random Forest)
│   └── acquisition.py  # Exploration strategies (EI, UCB, Thompson Sampling)
│
├── agents/             # Execution and Management Entities
│   ├── strategist.py   # Operates the surrogate model and selects actions
│   ├── builder.py      # Constructs 3D atomistic slabs (ASE/Pymatgen)
│   ├── compute.py      # Handles HPC execution and MLFF pre-screening
│   ├── evaluator.py    # Parses raw DFT outputs (e.g., DOSCAR electronic properties)
│   ├── memory.py       # Tracks the dataset and exploration graph
│   ├── governor.py     # Enforces objectives, budgets, and constraints
│   ├── dynamics.py     # Runs MLIP-driven relaxations and MD
│   └── mlip_manager.py # Manages ML Force Field training and inference
│
├── research/           # Autonomous Reasoning Layer
│   ├── hypothesis_agent.py # The PI Agent: Detects patterns, pivots objectives
│   ├── planner_agent.py    # The Research Planner: Converts hypotheses to Campaigns
│   ├── experiment_graph.py # Semantic Knowledge Graph of the campaign
│   └── theory_builder.py   # Synthesizes data into natural language theories
│
├── workflows/          # Orchestration Logic
│   ├── adsorption_campaign.py    # Standard BO loop for adsorption
│   └── mlip_active_learning.py   # High-throughput MLIP/DFT Active Learning loop
│
└── cli/                # Command-Line Interfaces
    ├── loop_cli.py     # Entry point to run campaigns (loads campaign.yaml)
    ├── builder_cli.py  # Utility to build physical structures from JSON states
    └── visualize_cli.py# Generates performance plots and exploration graphs
```

---

## 🎓 The Lab Metaphor: Roles & Responsibilities

| Role | Agent | Metaphor | Responsibility |
| :--- | :--- | :--- | :--- |
| **Agent 0** | **Hypothesis Agent** | **The Principal Investigator** | Analyzes the Knowledge Graph via RandomForest feature importance to form theories. |
| **Agent 0.5**| **Research Planner** | **The Research Scientist** | Translates high-level PI hypotheses into structured, executable `Campaign` plans. |
| **Agent 1** | **Research Governor** | **The Lab Manager** | Enforces the PI's current objective, budget, and experimental constraints. |
| **Agent 2** | **Optimization Strategist** | **The Senior Postdoc** | Operates the **Surrogate Model** and selects the optimal next experiment via Acquisition Functions. |
| **Agent 3** | **Structure Builder** | **The PhD Student** | Builds 3D atomic structures, enforcing charge compensation for aliovalent doping. |
| **Agent 4** | **Compute Manager** | **The Lab Technician** | Handles HPC Slurm submission, failure recovery, and MLFF pre-screening. |
| **Agent 5** | **Evaluator** | **The Data Analyst** | Parses raw DFT outputs (energies, DOS) and calculates complex thermodynamic/kinetic rewards. |
| **Agent 6** | **Memory Graph** | **The Lab Notebook** | A digital archive (Knowledge Graph) of states, transitions, and empirical results. |
| **Agent 7** | **MLIP Manager** | **The Theoretician** | Trains and deploys active-learning potentials to bypass slow DFT. |

---

## 📦 Installation & Configuration

1. **Install dependencies:**
   ```bash
   pip install .
   ```

2. **Define your campaign (`campaign.yaml`):**
   Externalize your research parameters to easily manage multiple campaigns.
   ```yaml
   objective:
     type: "adsorption_tuning"
     target_e_ads: -1.2
     adsorbate: "O"
   budget:
     max_evaluations: 200
   constraints:
     facet: [0, 0, 1]
     bulk: {"La": 0.5, "Sr": 0.5, "Mn": 1.0, "O": 3.0}
   active_learning:
     sigma_threshold: 0.1
   ```

## 🔁 Usage

### Start an Optimization Campaign
```bash
clasde-loop --config campaign.yaml
```

### Visualize the Discovery Graph
```bash
clasde-visualize --memory results/clasde_memory.json
```
Produces `clasde_summary.png` mapping the exploration trajectory and rewards.

### Build a Structure from Memory
```bash
clasde-builder my_state.json --output POSCAR
```

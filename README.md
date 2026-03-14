# CLASDE: Closed-Loop Autonomous Surface Discovery Engine

CLASDE is a multi-agent, autonomous optimization framework designed for the discovery of stable and high-performing surface configurations in complex functional materials and electrocatalysts.

The system is designed to mimic the roles of a world-class computational research group, automating the entire cycle from natural language conceptualization to high-fidelity HPC execution and physical law induction.

---

## Repository Structure

```text
CLASDE/
├── agents/             # DECISION MAKERS (The "Who")
│   ├── collaborator_agent.py # LLM interface for campaign formulation
│   ├── planner_agent.py      # Dynamic DAG generation (Workflow sequences)
│   ├── strategist_agent.py   # Bayesian candidate selection (BO)
│   ├── builder_agent.py      # Symmetry-aware structural construction
│   └── evaluator_agent.py    # Result interpretation & NIST-anchoring
│
├── science/            # DOMAIN OBJECTS (The "What")
│   ├── workflow_graph.py     # DAG execution engine (WorkflowExecutor)
│   ├── chemistry.py          # Data-driven cation site heuristics
│   ├── validator.py          # Physical constraint enforcement
│   ├── descriptors.py        # Band-center and GCN calculations
│   └── theory_builder.py     # Physical law discovery
│
├── memory/             # CENTRALIZED KNOWLEDGE
│   ├── knowledge_graph.py    # Semantic scientific provenance
│   └── experiment_db.py      # SQLite-backed experiment repository
│
├── execution/          # INFRASTRUCTURE (The "Action")
│   ├── compute_agent.py      # Backend abstraction (VASP, ASE, MLIP)
│   └── workflow_executor.py  # Orchestration of task dependencies
│
├── configs/            # CONFIGURATION & DATA
│   ├── default.yaml          # System-wide defaults
│   └── reference_data.yaml   # NIST-anchored thermochemical references
└── autonomous_watchdog.py # Persistence & recovery manager
```

---

## New High-Impact Features

### 1. **DAG-Based Workflow Execution**
CLASDE no longer relies on linear pipelines. The **Research Planner** generates a formal **Directed Acyclic Graph (DAG)** of tasks for each candidate. The **WorkflowExecutor** traverses this graph in topological order, ensuring that dependencies (e.g., `Build` $\to$ `Relax` $\to$ `Enumerate Sites`) are met and data is passed correctly between compute nodes.

### 2. **NIST-Anchored Adsorption Energies**
Adsorption energy calculations are now rigorously grounded. CLASDE utilizes a centralized `reference_data.yaml` derived from **NIST Gas-Phase Thermochemistry**. The **Evaluation Agent** prioritizes locally computed reference energies but falls back to these standard baseline values, ensuring that "Rewards" are physically meaningful and academically publishable.

### 3. **Generalized Perovskite Builder (Symmetry-Aware)**
The `StructureBuilder` has been upgraded to support distorted perovskites. It can now generate **Orthorhombic ($Pbnm$)** and **Tetragonal ($I4/mcm$)** slabs in addition to standard cubic systems, enabling accurate modeling of realistic functional oxides like $La_{1-x}Sr_xCo_{1-y}Fe_yO_3$ (LSCF).

### 4. **Data-Driven Chemistry Heuristics**
Hardcoded element checks have been removed. The `ChemistryPhysicist` now uses Pymatgen-integrated data (atomic radii, electronegativity, common oxidation states) to dynamically categorize cation sites and surface layer types for any arbitrary stoichiometry.

---

## Usage

### Natural Language Collaboration
Initiate a research project by describing your goal in plain English.
```bash
clasde-collaborator --prompt "I want to optimize the ORR activity on SrTiO3 by doping the B-site with transition metals."
```

### Direct Loop Execution
```bash
clasde-loop --config configs/test_sto_doping.yaml
```

---

## Installation & Configuration

1. **Install dependencies:**
   ```bash
   pip install .
   ```

2. **Compute Profile:**
   Configure `compute_profile.yaml` with your Slurm partition and VASP executable paths. Standard NIST references are provided in `configs/reference_data.yaml`.

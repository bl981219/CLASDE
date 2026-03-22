# CLASDE: Closed-Loop Autonomous Surface Discovery Engine

<p align="center">
  <img src="docs/assets/CLASDE.png" width="800" title="CLASDE Architecture">
</p>

CLASDE is a multi-agent framework designed to automate the discovery of stable and high-performing surface configurations in complex functional materials. The engine mimics the hierarchy of a computational research group, integrating Large Language Models (LLMs) for conceptualization with high-performance computing (HPC) for physical execution.

---

## Repository Architecture

The system is organized into distinct layers to separate scientific reasoning from computational execution.

```text
CLASDE/
├── agents/             # DECISION MAKERS (The "Who")
│   ├── collaborator_agent.py # Human-Machine Interface (LLM)
│   ├── hypothesis_agent.py   # Scientific Theory Induction (PI)
│   ├── planner_agent.py      # Task Sequence Formulation
│   ├── governor_agent.py     # Budget & Constraint Enforcement (Lab Manager)
│   ├── strategist_agent.py   # Experiment Selection (BO / Senior Postdoc)
│   ├── builder_agent.py      # Symmetry-aware structural construction
│   └── evaluator_agent.py    # Result Interpretation (Data Analyst)
│
├── core/               # SCIENTIFIC PRIMITIVES
│   ├── campaign_manager.py   # Central Loop Orchestrator
│   ├── state.py              # SurfaceState (Source of Truth)
│   ├── action.py             # Mutation operators
│   ├── transition.py         # Physics rules
│   └── workflow_graph.py     # DAG Engine & WorkflowExecutor
│
├── science/            # DOMAIN OBJECTS (The "What")
│   ├── chemistry.py          # Data-driven cation site heuristics
│   ├── validator.py          # Physical constraint enforcement
│   ├── descriptors.py        # Band-center and GCN calculations
│   ├── theory_builder.py     # Physical law discovery
│   ├── adsorption_site_finder.py # Symmetry-unique site detection
│   └── reaction_network.py   # Microkinetic modelling structures
│
├── memory/             # CENTRALIZED KNOWLEDGE
│   ├── knowledge_graph.py    # Semantic scientific provenance
│   ├── experiment_db.py      # SQLite-backed experiment repository
│   ├── hypothesis_db.py      # Database of PI theories
│   ├── literature_db.py      # Prior knowledge storage
│   └── storage_provider.py   # Backend-agnostic Storage Registry
│
├── execution/          # INFRASTRUCTURE (The "Action")
│   ├── compute_agent.py      # Backend abstraction (VASP, ASE, MLIP)
│   └── auth_provider.py      # Modular credential management
│
├── workflows/          # ORCHESTRATION (The "Process")
│   ├── neb_workflow.py        # Transition state sequences
│   └── templates/             # Reusable task sequences
│
├── configs/            # CONFIGURATION & DATA
└── autonomous_watchdog.py # Persistence & recovery manager
```

### The Lab Metaphor: Roles and Responsibilities

| Role | Metaphor | Responsibility |
| :--- | :--- | :--- |
| **Strategic Collaborator** | **The Investor/Expert** | Translates natural language intent into formal scientific campaigns. |
| **Principal Investigator** | **The PI Agent** | The logic lead. Formulates testable hypotheses and verifies them against empirical data. |
| **Research Planner** | **The Architect** | Translates the PI's hypothesis into machine-runnable Directed Acyclic Graphs (DAGs). |
| **Research Governor** | **The Lab Manager** | Enforces objectives, hard budget safety ceilings, and chemical constraints (e.g. charge neutrality). |
| **Optimization Strategist** | **The Senior Postdoc** | Operates the Surrogate Model and selects the optimal next experiment via Bayesian Optimization. |
| **Structure Builder** | **The PhD Student** | Builds 3D atomic structures, enforcing symmetry-aware distortions and stoichiometric constraints. |
| **Compute Manager** | **The Lab Technician** | Orchestrates HPC execution (VASP, MLIP) with backend abstraction and autonomous recovery. |
| **Evaluation Agent** | **The Data Analyst** | Parses raw outputs and anchors calculated rewards to NIST-benchmarked thermochemical data. |

---

## How CLASDE Works: The Discovery Loop

CLASDE operates through a self-correcting feedback loop centered on the Principal Investigator (PI). This loop elevates the system from simple optimization to autonomous scientific discovery.

```mermaid
graph TD
    User((User Intent)) -->|Natural Language| Agent1[Strategic Collaborator]
    Agent1 -->|Campaign Config| Agent2[Research Governor]
    
    subgraph Autonomous_Loop [The Hypothesis-Driven Loop]
        Agent2 -->|Budget & Constraints| Agent3[Optimization Strategist]
        Agent3 -->|Observation| Memory[(Knowledge Graph)]
        Memory -->|Prior Data| Agent3
        Agent3 -->|Candidate Surface| Agent4[Research Planner]
        Agent5[Principal Investigator] -->|Active Hypothesis| Agent4
        Agent4 -->|Structured DAG| Agent6[Workflow Executor]
        Agent6 -->|HPC Execution| VASP[VASP/ASE Backend]
        VASP -->|Raw Outputs| Agent7[Evaluation Agent]
        Agent7 -->|Physical Observables| Memory
        Memory -->|Trends & Patterns| Agent5
        Agent5 -->|Verification| Memory
        Agent5 -->|Evolved Hypothesis| Agent4
    end
    
    Memory -->|Discovery Report| Output((Scientific Insights))
```

### Implementation Logic
1. **Literature Ingestion**: The engine identifies relevant scientific claims from the local `LiteratureDatabase`.
2. **Hypothesis Formulation**: The **Principal Investigator (PI)** Agent synthesizes these claims into a formal Scientific Hypothesis.
3. **Strategic Selection**: The **Strategist** uses Bayesian Optimization to identify the next surface configuration that most effectively tests the current belief state.
4. **DAG Planning**: The **Research Planner** translates the PI's hypothesis and the Strategist's selected action into a formal Directed Acyclic Graph (DAG) of tasks.
5. **Autonomous Execution**: The **Campaign Manager** orchestrates the loop, delegating task sequencing to the Planner and execution to the Workflow Executor.
6. **Verification & Evolution**: The **PI** reviews the findings, verifies or falsifies the current theory, and evolves the hypothesis for the next cycle.

---

## Installation and Environment Setup

### 1. Python Environment
The engine requires Python 3.9 or higher. It is recommended to use a virtual environment or Conda.

```bash
# Development Installation
git clone <repository-url>
cd clasde_bill
pip install -e .
```

### 2. Dependency Management
Core dependencies are managed via `pyproject.toml`. Key packages include:
- **Pymatgen**: Primary library for structural analysis and symmetry detection.
- **ASE (Atomic Simulation Environment)**: Interface for interatomic potentials and calculator management.
- **CHGNet**: Machine-learned interatomic potential for rapid screening.
- **Scikit-Learn**: Provides the Gaussian Process surrogates for the Strategist.

---

## Configuration Guide

### 1. API Credentials (.env)
The system requires a Google Gemini API key for natural language reasoning and optional HPC credentials for cluster execution. Create a `.env` file in the root directory:

```text
GOOGLE_API_KEY=your_api_key_here
CLASDE_MOCK_LLM=false

# Optional HPC Credentials
HPC_USER=your_username
HPC_HOST=your_cluster_host
HPC_KEY_PATH=/path/to/your/ssh/key
```

### 2. Compute Profile (compute_profile.yaml)
This file defines the interface between the engine and your specific computing environment. It must be present in the root or a configured path.

```yaml
platform: "hpc" # Options: hpc, local
executable: "/path/to/vasp_std"
run_command: "mpirun -np {ntasks} {executable}"

# Slurm Header Configuration
slurm:
  partition: "xeon-p8"
  extra_header: |
    #SBATCH --time=24:00:00
    module load intel-oneapi/2023.1
    source /etc/profile

# Default VASP Parameters
vasp_params:
  PREC: "Accurate"
  ENCUT: 450
  LORBIT: 11
```

### 3. Reference Data (reference_data.yaml)
Adsorption energies are calculated relative to gas-phase species. This file contains standard NIST-anchored baselines used if computed references are not yet available in the local database.

---

## Operational Commands

### Natural Language Research
Use the Collaborator CLI to start a campaign from a research question.
```bash
clasde-collaborator --prompt "How does sulfur poisoning affect the oxygen reduction reaction on LSCF?"
```

### Long-Running Persistence (Watchdog)
For campaigns executed on clusters with potential login node timeouts, use the standalone watchdog to maintain campaign health and handle Slurm re-attachment.
```bash
python3 autonomous_watchdog.py --configs configs/test_lscf_poisoning.yaml
```

### Direct Execution
Launch a pre-configured YAML campaign directly.
```bash
clasde-loop --config configs/your_campaign.yaml
```

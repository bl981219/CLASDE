# CLASDE: Closed-Loop Autonomous Surface Discovery Engine

CLASDE is a mathematically defined, multi-agent, autonomous optimization framework designed for the discovery of stable, active, and high-performing surface configurations in complex functional materials and electrocatalysts.

---

## Key Features
- **Natural Language Collaboration:** A Strategic Collaborator Agent translates high-level research questions into structured optimization campaigns using Large Language Models (LLMs).
- **Hypothesis-Driven Discovery:** A Principal Investigator (PI) Agent analyzes experimental data graphs to formulate physical hypotheses and propose new research directions autonomously.
- **Active Learning via MLIPs:** Continuous 24/7 exploration using Machine Learning Interatomic Potentials (MLIPs) with automated uncertainty-triggered DFT corrections.
- **Segregation Modeling:** Specialized rewards and action operators (e.g., SWAP_ATOMS) for investigating surface enrichment and thermodynamic stability of dopants.
- **Hard Budget Ceilings:** Global safety limits that prevent autonomous agents from exceeding predefined computational resource allocations.
- **HPC Robustness:** Multi-mode Compute Manager that handles Slurm job submission, queue monitoring (squeue), and automatic recovery from common DFT failures.

---

## Repository Structure

The framework separates physical ground truths, optimization mathematics, execution agents, and autonomous reasoning.

```text
CLASDE/
├── core/               # Scientific Primitives
│   ├── state.py        # Canonical SurfaceState representation (Pydantic)
│   ├── campaign.py     # Formal representation of a scientific research campaign
│   ├── action.py       # Formal mutation operators (vacancies, swaps, etc.)
│   ├── transition.py   # Deterministic application of actions to states
│   └── reward.py       # Reward functions (Segregation, Phase Diagram, Discovery)
│
├── optimization/       # Bayesian Optimization Engine
│   ├── surrogate.py    # Probabilistic models (GPR, Random Forest)
│   └── acquisition.py  # Exploration strategies (EI, UCB, Thompson Sampling)
│
├── agents/             # Execution and Management Entities
│   ├── collaborator.py # LLM-driven Natural Language interface (Agent -1)
│   ├── strategist.py   # Operates the surrogate model and selects actions
│   ├── builder.py      # Constructs 3D atomistic slabs (ASE/Pymatgen)
│   ├── compute.py      # Handles HPC execution and MLFF pre-screening
│   ├── evaluator.py    # Parses raw DFT outputs
│   ├── memory.py       # Tracks the dataset and exploration graph
│   ├── governor.py     # Enforces objectives, budgets, and safety ceilings
│   ├── dynamics.py     # Runs MLIP-driven relaxations and MD
│   └── mlip_manager.py # Manages ML Force Field training and inference
│
├── research/           # Autonomous Reasoning Layer
│   ├── hypothesis_agent.py # The PI Agent: Detects patterns, pivots objectives
│   ├── planner_agent.py    # The Research Planner
│   ├── experiment_graph.py # Semantic Knowledge Graph
│   └── theory_builder.py   # Synthesizes data into natural language theories
│
├── workflows/          # Orchestration Logic
│   ├── autonomous_campaign.py    # General BO loop for optimization
│   └── mlip_active_learning.py   # High-throughput MLIP/DFT loop
│
├── teaching/           # Educational Demos and System Verification
│   ├── teaching_demo.py # Guided walkthrough of the autonomous loop
│   └── teaching_results/# Preserved logs for educational purposes
│
└── cli/                # Command-Line Interfaces
    ├── collaborator_cli.py # Natural Language research interface
    ├── loop_cli.py         # Entry point to run campaigns from YAML
    ├── builder_cli.py      # Utility to build physical structures
    └── visualize_cli.py    # Generates plots and discovery graphs
```

---

## The Lab Metaphor: Roles & Responsibilities

| Role | Agent | Metaphor | Responsibility |
| :--- | :--- | :--- | :--- |
| **Agent -1**| **Collaborator** | **The Investor/Expert** | Translates natural language intent into formal scientific campaigns via LLMs. |
| **Agent 0** | **Hypothesis Agent** | **The Principal Investigator** | Analyzes the Knowledge Graph to form scientific theories. |
| **Agent 1** | **Research Governor** | **The Lab Manager** | Enforces objectives, hard budget safety ceilings, and constraints. |
| **Agent 2** | **Optimization Strategist** | **The Senior Postdoc** | Operates the Surrogate Model and selects optimal next experiments. |
| **Agent 3** | **Structure Builder** | **The PhD Student** | Builds 3D atomic structures, enforcing charge compensation and segregation logic. |
| **Agent 4** | **Compute Manager** | **The Lab Technician** | Handles HPC execution, failure recovery, and MLFF pre-screening. |
| **Agent 5** | **Evaluator** | **The Data Analyst** | Parses raw DFT outputs and calculates scalar rewards. |
| **Agent 6** | **Memory Graph** | **The Lab Notebook** | A digital archive (Knowledge Graph) of states and transitions. |

---

## Installation & Configuration

1. **Install dependencies:**
   ```bash
   pip install .
   ```

2. **Configure API Access:**
   To enable the Natural Language interface and the Strategic Collaborator Agent, a Google Gemini API key is **required**:
   - Visit [Google AI Studio](https://aistudio.google.com/) to generate your key.
   - Copy the example environment file:
     ```bash
     cp .env_example .env
     ```
   - Open `.env` and paste your key into the `GOOGLE_API_KEY` field.
   
   *Note: Your `.env` file is automatically ignored by Git to protect your sensitive credentials.*

3. **Configure HPC Templates (Optional):**
   CLASDE can autonomously detect your Slurm environment and generate submission scripts. However, for specific cluster configurations:
   - Place your custom VASP Slurm script at `workflows/templates/slurm_vasp.sh`.
   - Use the placeholder `#SBATCH -J name` to allow the agent to name jobs automatically.
   - If no template is found, the **Compute Manager** will probe your system (via `sinfo`) and generate a compatible script by itself.

## Usage

### Natural Language Collaboration
Initiate a campaign by describing your research goal in plain English.
```bash
clasde-collaborator --prompt "how does Sr segregation in LaSrFeO3 depend on T and PO2?"
```
The system will propose a strategy and require your confirmation before starting the autonomous loop.

### Start a Campaign from YAML
```bash
clasde-loop --config configs/default.yaml
```

### Visualize the Discovery Graph
```bash
clasde-visualize --memory data/results/clasde_memory.json
```
Produces summary plots mapping the exploration trajectory and rewards.

---

## Safety and Resource Management
CLASDE implements hard budget ceilings (default 100 evaluations) to prevent runaway computational costs. Any campaign configuration requesting more than this limit will be automatically capped by the Research Governor.

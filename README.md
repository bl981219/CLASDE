# CLASDE: Closed-Loop Autonomous Surface Discovery Engine

CLASDE is a multi-agent, autonomous optimization framework designed for the discovery of stable and high-performing surface configurations. 

Following expert architectural review, the repository is organized into a hierarchy that separates decision-makers from domain objects.

---

## Repository Structure

```text
CLASDE/
├── agents/             # DECISION MAKERS (The "Who")
│   ├── collaborator_agent.py # Agent -1: Human-Machine Interface (LLM)
│   ├── hypothesis_agent.py   # Agent 0: Scientific Theory Induction
│   ├── planner_agent.py      # Agent 0.5: Campaign Formulation
│   ├── governor_agent.py     # Agent 1: Budget & Constraint Enforcement
│   ├── strategist_agent.py   # Agent 2: Experiment Selection (BO)
│   ├── builder_agent.py      # Agent 3: Structural Construction
│   └── evaluator_agent.py    # Agent 5: Result Interpretation
│
├── science/            # DOMAIN OBJECTS (The "What")
│   ├── experiment_graph.py   # Semantic Knowledge Graph
│   ├── hypothesis.py         # Scientific Uncertainty Modeling
│   └── theory_builder.py     # Natural Language Theory Synthesis
│
├── memory/             # CENTRALIZED KNOWLEDGE (The "Where")
│   ├── knowledge_graph.py    # Persistence for cross-campaign logic
│   ├── experiment_memory.py  # Local trajectory storage
│   └── literature_memory.py  # Prior knowledge & Literature ingestion
│
├── optimization/       # MATHEMATICS (The "How")
│   ├── surrogate_models.py   # GPR, Random Forest, etc.
│   ├── acquisition_functions.py # EI, UCB, Thompson Sampling
│   └── campaign_optimizer.py # BO Orchestration
│
├── execution/          # INFRASTRUCTURE (The "Action")
│   ├── compute_agent.py      # HPC/Slurm Execution (Agent 4)
│   ├── mlip_manager.py       # Force Field management
│   ├── dynamics_engine.py    # Relaxation & MD
│   └── workflow_runner.py    # Main autonomous loop
│
├── core/               # SCIENTIFIC PRIMITIVES
│   ├── state.py              # SurfaceState representation
│   ├── action.py             # Mutation operators
│   ├── transition.py         # Physics rules
│   └── reward.py             # Objective functions
│
├── cli/                # Command-Line Interfaces
└── teaching/           # Educational Demos
```

---

---

## The Lab Metaphor: Roles & Responsibilities

CLASDE mimics the hierarchy of a world-class computational research group, where specialized agents collaborate through a shared knowledge base.

| Role | Agent | Responsibility | Metaphor |
| :--- | :--- | :--- | :--- |
| **Strategic Collaborator** | `Agent -1` | Translates natural language intent into formal campaigns. | **The Investor/Expert** |
| **Hypothesis Agent** | `Agent 0` | Induces physical laws from the Knowledge Graph. | **The Principal Investigator** |
| **Research Planner** | `Agent 0.5` | Dynamically constructs task sequences (MD, NEB, Relax). | **The Research Planner** |
| **Research Governor** | `Agent 1` | Enforces budget ceilings and chemical constraints. | **The Lab Manager** |
| **Optimization Strategist** | `Agent 2` | Operates surrogate models to select next experiments. | **The Senior Postdoc** |
| **Structure Builder** | `Agent 3` | Constructs 3D atomistic slabs and applies defects. | **The PhD Student** |
| **Compute Manager** | `Agent 4` | Orchestrates HPC execution and failure recovery. | **The Lab Technician** |
| **Evaluation Agent** | `Agent 5` | Parses raw DFT/MLFF outputs into scalar rewards. | **The Data Analyst** |

---

## How CLASDE Works: The Agentic Discovery Loop

The system operates as a continuous, self-correcting feedback loop centered around a **Scientific Knowledge Graph**.

```text
[ User Intent ] 
      |
      v
[ Collaborator Agent ]  <--- (Natural Language -> Formal Campaign)
      |
      +----------------------------+
      |                            |
      v                            v
[ Research Governor ]       [ Hypothesis Agent ] <--- (Theory Induction)
      |                            ^
      v                            |
[ Strategist Agent ] <------- [ Knowledge Graph ] <--- (Central Memory)
      |                            ^
      +----(Propose Actions)-------+
      |                            |
      v                            |
[ Planner Agent ]  <---------(Observe Beliefs)
      |
      v (Dynamic Workflow: MD -> Relax -> Adsorption)
      |
[ Builder Agent ]  -----> [ Compute Manager ] -----> [ Evaluation Agent ]
      |                          |                          |
      +---(Build POSCAR)         +---(Run VASP/MLIP)        +---(Extract R)
```

1.  **Observing State:** The Strategist observes the current `ExperimentDatabase` and the `KnowledgeGraph`.
2.  **Updating Belief:** The internal Surrogate Model (Gaussian Process) is refitted with new data.
3.  **Proposing Actions:** The Strategist projects potential mutations (vacancies, swaps, coverage changes).
4.  **Dynamic Planning:** The Planner generates a custom task sequence for the most promising candidates.
5.  **Autonomous Execution:** The Compute Manager probes the cluster, submits Slurm jobs, and handles restarts.
6.  **Knowledge Integration:** Results are decomposed into semantic nodes and relations, updating the group's "Scientific Memory."

---

## Key Features
- **Semantic Knowledge Graph:** Tracks Material -> Surface -> Structure -> Result with full provenance.
- **Dynamic Workflows:** Agents decide if a structure needs MD pre-equilibration or NEB barrier mapping.
- **HPC Robustness:** Autonomous Slurm management with automatic SCF/Ionic recovery.
- **Multi-Objective Optimization:** Acquisition functions balance Reward, Uncertainty, Novelty, and Cost.
- **Scientific Uncertainty:** Quantifies the epistemic support for every discovered physical law.

---

## Installation & Configuration

1. **Install dependencies:**
   ```bash
   pip install .
   ```

2. **Run Tests (Optional):**
   ```bash
   python -m unittest discover tests
   ```

3. **Configure API Access:**
   Copy `.env_example` to `.env` and add your Google Gemini API key.

## Usage

### Natural Language Collaboration
```bash
clasde-collaborator --prompt "how does Sr segregation in LaSrFeO3 depend on T?"
```

### Start a Campaign from YAML
```bash
clasde-loop --config configs/default.yaml
```

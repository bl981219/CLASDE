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

CLASDE mimics the hierarchy of a world-class computational surface science group. The system is designed not as a generic optimizer, but specifically to discover catalytic mechanisms, adsorption scaling relations, and stable surface phases.

| Role | Agent | Responsibility | Metaphor |
| :--- | :--- | :--- | :--- |
| **Strategic Collaborator** | `Agent -1` | Translates natural language intent into formal surface science campaigns (e.g., "Find CO oxidation pathways on Pt"). | **The Investor/Expert** |
| **Hypothesis Agent** | `Agent 0` | Induces physical laws (e.g., d-band center correlations, scaling relations) from the Knowledge Graph. | **The Principal Investigator** |
| **Research Planner** | `Agent 0.5` | Dynamically constructs task sequences based on scientific reasoning (e.g., if unstable -> run MD; if pathway unknown -> run NEB). | **The Research Planner** |
| **Research Governor** | `Agent 1` | Enforces budget ceilings, Sabatier optimum windows, and chemical constraints. | **The Lab Manager** |
| **Optimization Strategist** | `Agent 2` | Operates surrogate models to balance Expected Reward, Uncertainty, Novelty, and Cost. | **The Senior Postdoc** |
| **Structure Builder** | `Agent 3` | Constructs 3D atomistic slabs, places specific adsorbates on defined sites (top, bridge, hollow), and manages coverages. | **The PhD Student** |
| **Compute Manager** | `Agent 4` | Orchestrates HPC execution (VASP, MLIP, MD, NEB) and handles SCF/Ionic failure recovery. | **The Lab Technician** |
| **Evaluation Agent** | `Agent 5` | Parses raw DFT outputs into core surface metrics (Adsorption Energy, Reaction Barrier, d-band center, Work Function). | **The Data Analyst** |

---

## How CLASDE Works: Surface Science Agentic Loop

The system operates as a continuous, self-correcting feedback loop centered around a **Scientific Knowledge Graph** tailored for surface catalysis.

```text
===================================================================================
                     THE CLASDE AUTONOMOUS DISCOVERY ENGINE
===================================================================================

[ USER INTENT ]  "Explore CO adsorption on Cu(111)"
      |
      v
[ COLLABORATOR AGENT ] -----> Formulates Objective: Sabatier Target E_ads = -1.5 eV
      |
      +-------------------------------------------------------------------+
      |                                                                   |
      v                                                                   v
[ RESEARCH GOVERNOR ]                                           [ HYPOTHESIS AGENT ]
(Enforces max 100 DFT calls)                                    (Induces Scaling Relations)
      |                                                                   ^
      v                                                                   |
[ STRATEGIST AGENT ] <----------------------------------------- [ KNOWLEDGE GRAPH ]
(Surrogate Model + UCB)                                           (Central Memory)
      |                                                                   ^
      +----(Proposes: Add Vacancy, Change Coverage)-----------------------+
      |                                                                   |
      v                                                                   |
[ PLANNER AGENT ]  <-----(Scientific Reasoning: High Uncertainty?)        |
      |                                                                   |
      v (Dynamic Workflow: MLIP Relax -> NEB Barrier -> DFT Adsorption)   |
      |                                                                   |
[ BUILDER AGENT ] ---------> [ COMPUTE MANAGER ] ---------> [ EVALUATION AGENT ]
(Slab/Site Gen)              (VASP / MACE / MD)             (Extracts d-band, E_a)
```

1.  **State Representation:** Everything revolves around the `SurfaceState`, tracking bulk composition, facet, termination, specific adsorbate sites/orientations, coverage, defects, and strain.
2.  **Scientific Planning:** The Planner agent doesn't use hardcoded pipelines. If an adsorption energy is highly uncertain, it requests more adsorption calculations. If a surface is unstable, it triggers MD. If a mechanism is unknown, it triggers NEB.
3.  **Execution & Recovery:** The Compute Manager uses specialized tools (`slab_generator`, `adsorption_site_finder`, `neb_runner`) to submit jobs to Slurm, actively monitoring and fixing divergence issues.
4.  **Knowledge Integration:** Results update the `KnowledgeGraph`, linking `Surface -> Site -> Adsorbate -> Reaction Path`.

---

## Key Features
- **Surface Science Ontology:** Native support for modeling reaction pathways, activation barriers, surface reconstructions, and coverage effects.
- **Dynamic Workflows:** Agents autonomously decide the execution path (e.g., MD pre-equilibration vs. NEB barrier mapping).
- **HPC Robustness:** Autonomous Slurm management with automatic SCF/Ionic recovery.
- **Multi-Objective Optimization:** Acquisition functions balance Catalytic Activity, Uncertainty, Novelty, and Computational Cost.
- **Scientific Uncertainty:** Quantifies the epistemic support for every discovered physical law (e.g., d-band theory).

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

### Domain-Specific Surface Exploration
Launch a targeted surface science campaign directly from the CLI:
```bash
clasde-explore LaSrFeO3 001 O
```

### Natural Language Collaboration
Initiate a campaign by describing your research goal in plain English:
```bash
clasde-collaborator --prompt "how does Sr segregation in LaSrFeO3 depend on T?"
```

### Start a Campaign from YAML
```bash
clasde-loop --config configs/default.yaml
```

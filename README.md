# CLASDE: Closed-Loop Autonomous Surface Discovery Engine

CLASDE is a multi-agent, autonomous optimization framework designed for the discovery of stable and high-performing surface configurations in complex functional materials and electrocatalysts.

Following expert architectural review, the repository is organized into a hierarchy that separates decision-makers from domain objects.

---

## Repository Structure

```text
CLASDE/
├── agents/             # DECISION MAKERS (The "Who")
│   ├── collaborator_agent.py # Human-Machine Interface (LLM)
│   ├── hypothesis_agent.py   # Scientific Theory Induction (PI)
│   ├── planner_agent.py      # Dynamic Workflow Formulation
│   ├── governor_agent.py     # Budget & Constraint Enforcement
│   ├── strategist_agent.py   # Experiment Selection (BO)
│   ├── builder_agent.py      # Structural Construction
│   └── evaluator_agent.py    # Result Interpretation
│
├── science/            # DOMAIN OBJECTS (The "What")
│   ├── experiment_graph.py   # Semantic Knowledge Graph
│   ├── hypothesis.py         # Scientific Uncertainty Modeling
│   ├── objective_functions.py# Sabatier and Catalytic Metrics
│   ├── reaction_network.py   # Catalytic Cycles & Reaction Pathways
│   ├── descriptors.py        # d-band, Coordination, Bader charges
│   └── theory_builder.py     # Natural Language Theory Synthesis
│
├── memory/             # CENTRALIZED KNOWLEDGE (The "Where")
│   ├── knowledge_graph.py    # Persistence for cross-campaign logic
│   ├── experiment_db.py      # Detailed physical/computational database
│   ├── hypothesis_db.py      # Formal scientific theory storage
│   └── literature_db.py      # Prior knowledge & Literature ingestion
│
├── optimization/       # MATHEMATICS (The "How")
│   ├── surrogate_models.py   # GPR, Random Forest, etc.
│   ├── acquisition_functions.py # EI, UCB, Thompson Sampling
│   └── campaign_optimizer.py # BO Orchestration
│
├── execution/          # INFRASTRUCTURE (The "Action")
│   ├── compute_agent.py      # HPC/Slurm Execution
│   ├── mlip_manager.py       # Force Field management
│   ├── dynamics_engine.py    # Relaxation & MD
│   ├── neb_runner.py         # Transition State search (NEB)
│   ├── slab_generator.py     # Surface cleaving
│   ├── adsorption_site_finder.py # High-symmetry site detection
│   ├── coverage_generator.py # Lateral interaction modeling
│   └── workflow_runner.py    # Main autonomous loop
│
├── core/               # SCIENTIFIC PRIMITIVES
│   ├── state.py              # SurfaceState representation
│   ├── action.py             # Mutation operators
│   └── transition.py         # Physics rules
│
├── cli/                # Command-Line Interfaces
└── examples/           # Educational Demos and Tutorials
```

---

## The Lab Metaphor: Roles & Responsibilities

CLASDE mimics the hierarchy of a world-class computational surface science group. The system is designed not as a generic optimizer, but specifically to discover catalytic mechanisms, adsorption scaling relations, and stable surface phases.

| Role | Responsibility | Metaphor |
| :--- | :--- | :--- |
| **Strategic Collaborator** | Translates natural language intent into formal surface science campaigns (e.g., "Find CO oxidation pathways on Pt"). | **The Investor/Expert** |
| **Principal Investigator** | Induces physical laws (e.g., d-band center correlations, scaling relations) from the Knowledge Graph. | **The PI Agent** |
| **Research Planner** | Dynamically constructs task sequences based on scientific reasoning (e.g., if unstable -> run MD; if pathway unknown -> run NEB). | **The Planner** |
| **Research Governor** | Enforces budget ceilings, Sabatier optimum windows, and chemical constraints. | **The Lab Manager** |
| **Optimization Strategist** | Operates surrogate models to balance Expected Reward, Uncertainty, Novelty, and Cost. | **The Senior Postdoc** |
| **Structure Builder** | Constructs 3D atomistic slabs, places specific adsorbates on defined sites (top, bridge, hollow), and manages coverages. | **The PhD Student** |
| **Compute Manager** | Orchestrates HPC execution (VASP, MLIP, MD, NEB) and handles SCF/Ionic failure recovery. | **The Lab Technician** |
| **Evaluation Agent** | Parses raw DFT outputs into core surface metrics (Adsorption Energy, Reaction Barrier, d-band center, Work Function). | **The Data Analyst** |

---

## Research Modes: Mapping, Tuning, and Stability

CLASDE allows you to toggle between three fundamental modes of research by setting the `research_mode` field in your campaign configuration. This ensures that exploration and optimization are treated as connected strategies.

| Mode | Scientific Intent | Agent Behavior | Use Case |
| :--- | :--- | :--- | :--- |
| **MAPPING** | **Pure Discovery** | Maximizes **Uncertainty & Novelty**. The goal is to build an accurate physical model of the entire space. | "How does $SO_2$ affect LSCF across different facets and temperatures?" |
| **TUNING** | **Optimization** | Maximizes **Expected Improvement**. Standard Bayesian Optimization focused on finding the best material. | "Find the dopant that minimizes the oxygen adsorption energy." |
| **STABILITY** | **Thermodynamics** | Minimizes **Grand Potential**. Focuses on finding the most stable phase under varying $(T, P)$ conditions. | "What is the equilibrium surface structure of LSF at 1000 K?" |

---

## How CLASDE Works: The Agentic Discovery Loop

CLASDE operates through a self-correcting feedback loop where specialized agents interact via a shared **Scientific Knowledge Graph**. This loop elevates the system from simple "search" to "autonomous discovery."

### 1. Conceptualization (Natural Language to Formal Goal)
The discovery starts when a user provides a research question. The **Collaborator Agent** translates this intent into a formal **Campaign** by selecting the appropriate **Research Mode**. For example, starting with $LaSrFeO_3$ (LSF):
- **Question:** *"How does oxygen adsorption change across all facets of LSF?"* -> **MAPPING Mode** (Builds a global electronic property map).
- **Question:** *"Which dopant minimizes the oxygen vacancy formation energy on LSF?"* -> **TUNING Mode** (Finds the optimal chemistry).
- **Question:** *"What is the equilibrium structure of the LSF surface at 1000 K and 1 atm O2?"* -> **STABILITY Mode** (Finds the thermodynamic global minimum).

### 2. Strategic Observation (Memory to Belief)
The **Optimization Strategist** observes all prior experiments stored in the **Knowledge Graph**. It updates its internal **Belief State**—a probabilistic surrogate model (Gaussian Process)—that maps structural descriptors to physical performance.

### 3. Hypothesis Generation (PI Reasoning)
Simultaneously, the **Principal Investigator (PI)** agent analyzes the graph for emergent trends. It calculates statistical support for physical laws (e.g., "Is d-band center a valid predictor for this surface?"). These induced theories are used to bias the search toward scientifically interesting regions.

### 4. Dynamic Planning (Task Sequencing)
Unlike static pipelines, the **Research Planner** dynamically generates a sequence of tasks for each candidate structure. If the PI is uncertain about stability, the Planner might insert a Molecular Dynamics (MD) equilibration step before the final DFT relaxation.

### 5. Physical Execution (HPC Orchestration)
The **Compute Manager** translates these plans into HPC job scripts. It probes the cluster environment, submits to Slurm, and monitors the queue. If a calculation diverges (e.g., electronic SCF failure), the agent autonomously applies a physical fix and restarts the job.

### 6. Knowledge Integration (The Digital Lab Notebook)
Finally, the **Evaluation Agent** parses the raw output files. Results are not just saved as numbers; they are decomposed into semantic nodes (Sites, Intermediates, Transitions) and integrated back into the **Knowledge Graph**, completing the discovery cycle.

#### Agent Communication & Data Flow
The following scheme illustrates how agents interact and pass messages within the discovery loop:

```text
================================================================================================
                            CLASDE AGENTIC MESSAGE-PASSING SCHEME
================================================================================================

[ USER ] --------------------( Research Question )--------------------> [ COLLABORATOR AGENT ]
                                                                                 |
                                                                         ( Campaign Config )
                                                                                 |
                                                                                 v
+----------------------------------------------------------------------- [ RESEARCH GOVERNOR ]
|                                                                                |
|                                                                        ( Constraints/Budget )
|                                                                                |
v                                                                                v
[ HYPOTHESIS AGENT ] <-------( Scientific Provenance )---------+-------> [ STRATEGIST AGENT ]
         |                                                     |                 |
  ( Physical Laws )                                     [ KNOWLEDGE GRAPH ]      ( Optimal State )
         |                                              ( Central Memory )       |
         v                                                     ^                 v
[ PLANNER AGENT ] <--------------------------------------------+---------- ( Observe Beliefs )
         |
  ( Dynamic Task Sequence: e.g., MD -> Relax -> NEB )
         |
         v
[ BUILDER AGENT ] ---------( POSCAR )--------> [ COMPUTE MANAGER ] --------( Raw Output )------>
         |                                             |                                       |
         +---------------------------------------------+---------------------------------------+
                                                       |
                                               [ EVALUATION AGENT ]
                                                       |
                                               ( Semantic Result )
                                                       |
                                                       v
                                              [ KNOWLEDGE GRAPH ]
```

---

## Key Features
- **Surface Science Ontology:** Native support for modeling reaction pathways, activation barriers, surface reconstructions, and coverage effects.
- **Advanced Descriptors:** Automated calculation and correlation analysis for **GCN**, **d-band center/edge**, **O2p center**, **charge transfer energy**, and **$e_g$ occupancy**.
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
Quickly launch a targeted campaign for a specific material, surface facet, and adsorbate. This command bypasses the natural language interface for direct execution.
```bash
# Syntax: clasde-explore <Material> <Facet> <Adsorbate>
clasde-explore LaSrFeO3 001 O
```
**Arguments:**
- **Material:** Chemical formula of the bulk substrate (e.g., `Cu`, `Pt`, `SrTiO3`, `LaSrFeO3`).
- **Facet:** Miller indices of the surface plane (e.g., `111`, `001`, `110`).
- **Adsorbate:** The chemical species to be adsorbed (e.g., `O`, `CO`, `OH`, `SO2`).

### Natural Language Collaboration
Initiate a research project by describing your goal in plain English. The **Collaborator Agent** will analyze your intent, suggest the optimal **Research Mode**, and formulate the full campaign configuration.
```bash
# Example: Start an interactive session or provide a direct prompt
clasde-collaborator --prompt "how does Sr segregation in LaSrFeO3 depend on temperature?"
```
**Options:**
- `--prompt`: Your research question or scientific goal. If omitted, the agent will start an interactive dialogue.
- `--key`: Manually provide a Google Gemini API key (alternatively, use the `.env` file).

### Start a Campaign from YAML
For full control, you can define your campaign in a standard YAML file and run it directly.
```bash
clasde-loop --config configs/default.yaml
```

---

## Case Studies & Examples

These case studies demonstrate how CLASDE can be used to investigate complex surface science problems autonomously.

### 1. Cr and S Poisoning on LSCF
An autonomous exploration of $CrO_3$ and $SO_2$ adsorption competition on the $La_{0.6}Sr_{0.4}Fe_{0.8}Co_{0.2}O_3$ (001) surface.
- **Scientific Context:** This campaign was inspired by and serves as an autonomous validation of research themes found in [DOI: 10.1021/acs.chemmater.4c01936](https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.4c01936).
- **Discovery Trajectory:** The agent autonomously identifies stable configurations and induces physical laws linking metal d-band centers to poison binding strength.
- **Location:** `examples/LSCF_Poisoning_CaseStudy/`

### 2. Sr Surface Segregation in LSF
A 50-iteration thermodynamic study mapping the Sr enrichment of $La_{0.6}Sr_{0.4}FeO_3$ as a function of Temperature and Oxygen Pressure.
- **Outcome:** Mapping the $(T, P_{O2})$ drivers for segregation and identifying the electronic stabilization of the segregated phase.
- **Location:** `examples/LSF_Segregation_CaseStudy/`

### 3. B-site Doping in SrTiO3
Autonomous screening of transition metal dopants (Mn, Fe, Co) to activate oxygen adsorption on the $SrTiO_3$ (001) surface.
- **Outcome:** Rapidly identified Mn as a high-performance dopant for enhancing surface reactivity.
- **Location:** `examples/SrTiO3_Doping_CaseStudy/`

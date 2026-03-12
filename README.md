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
│   ├── governor_agent.py     # Budget & Constraint Enforcement (Lab Manager)
│   ├── strategist_agent.py   # Experiment Selection (BO / Senior Postdoc)
│   ├── builder_agent.py      # Generalized Perovskite Construction
│   └── evaluator_agent.py    # Result Interpretation (Data Analyst)
│
├── science/            # DOMAIN OBJECTS (The "What")
│   ├── experiment_graph.py   # Semantic Knowledge Graph
│   ├── hypothesis.py         # Scientific Uncertainty Modeling
│   ├── objective_functions.py# Sabatier and Catalytic Metrics
│   ├── reaction_network.py   # Catalytic Cycles & Reaction Pathways
│   ├── descriptors.py        # d-band, Coordination, Bader charges
│   └── theory_builder.py     # Physical Law Discovery (Scaling, Bias)
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
│   ├── compute_agent.py      # HPC/Slurm Orchestration & Re-attachment
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
├── examples/           # Educational Demos and Test Prompts
└── autonomous_watchdog.py # Persistence & Multi-campaign monitor
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
| **Structure Builder** | Constructs generalized ABO3 perovskite slabs with dynamic termination detection and selective dynamics. | **The PhD Student** |
| **Compute Manager** | Orchestrates HPC execution (VASP, MLIP) with autonomous re-attachment and recovery. | **The Lab Technician** |
| **Evaluation Agent** | Parses raw DFT outputs into core surface metrics (Adsorption Energy, d-band center, O2p center). | **The Data Analyst** |

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

---

## Key Features
- **Generalized Perovskite Builder:** Native support for arbitrary $A_{1-x}A'_{x}B_{1-y}B'_{y}O_3$ stoichiometries with heuristic cation site assignment and automated selective dynamics.
- **Robust Electronic Analysis:** Deep integration with Pymatgen for reliable parsing of `vasprun.xml` to extract layer-resolved O 2p-band and d-band centers.
- **HPC Persistence (Watchdog):** Standalone watchdog process monitors active loops and re-attaches to existing Slurm jobs upon restart, ensuring zero data loss during timeouts.
- **Automated Theory Induction:** The `TheoryBuilder` automatically detects scaling relations, electronic descriptors, and termination-dependent reactivity biases.
- **Multi-Fidelity Workflows:** Seamlessly transitions between rapid MLIP (CHGNet/M3GNet) screening and high-accuracy VASP verification.

---

## Installation & Configuration

1. **Install dependencies:**
   ```bash
   pip install .
   ```

2. **Configure API Access:**
   Copy `.env_example` to `.env` and add your Google Gemini API key.

3. **Compute Profile:**
   Configure `compute_profile.yaml` with your Slurm partition and VASP executable paths.

---

## Usage

### Natural Language Collaboration
Initiate a research project by describing your goal in plain English.
```bash
# Example: Start an interactive session or provide a direct prompt
clasde-collaborator --prompt "I want to optimize the ORR activity on SrTiO3 by doping the B-site with transition metals."
```

### Multi-Campaign Monitoring (Watchdog)
To maintain persistence across multiple long-running campaigns:
```bash
clasde-watchdog --configs configs/test_lsf_segregation.yaml configs/test_sto_doping.yaml
```

### Direct Campaign Execution
```bash
clasde-loop --config configs/your_campaign.yaml
```

### Domain-Specific Surface Exploration
```bash
# Syntax: clasde-explore <Material> <Facet> <Adsorbate>
clasde-explore LaSrFeO3 001 O
```

---

## Case Studies & Examples

### 1. Cr and S Poisoning on LSCF
- **Location:** `examples/LSCF_Poisoning_CaseStudy/`
- **Scientific Goal:** Map the competition between SO2 and O adsorption on LSCF (001).
- **Scientific Context:** This campaign serves as an autonomous validation of research themes explored in [DOI: 10.1021/acs.chemmater.4c01936](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c01936), specifically focusing on how poisoning affects surface stability and electronic band centers.

### 2. Sr Surface Segregation in LSF
- **Location:** `examples/LSF_Segregation_CaseStudy/`
- **Scientific Goal:** Identify thermodynamic drivers for Sr enrichment in perovskite oxides.

### 3. B-site Doping in SrTiO3
- **Location:** `examples/SrTiO3_Doping_CaseStudy/`
- **Scientific Goal:** Rapidly screen transition metal dopants to activate the ORR.

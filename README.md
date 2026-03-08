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

## How CLASDE Works: The Agentic Discovery Loop

CLASDE operates through a self-correcting feedback loop where specialized agents interact via a shared **Scientific Knowledge Graph**. This loop elevates the system from simple "search" to "autonomous discovery."

### 1. Conceptualization (Natural Language to Formal Goal)
The discovery starts when a user provides a research question. The **Collaborator Agent** uses chemical domain knowledge to translate this intent into a formal **Campaign**. For example, "How does poisoning affect LSCF?" is translated into an objective to minimize adsorption energy for $SO_2$ on specific LSCF facets.

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
```bash
clasde-explore LaSrFeO3 001 O
```

### Natural Language Collaboration
```bash
clasde-collaborator --prompt "how does Sr segregation in LaSrFeO3 depend on T?"
```

---

## Case Studies & Examples

### Cr and S Poisoning on LSCF Perovskites
A full 100-iteration discovery campaign targeting $CrO_3$ and $SO_2$ adsorption competition on pristine $La_{0.6}Sr_{0.4}Fe_{0.8}Co_{0.2}O_3$.
- **Reference:** [DOI: 10.1021/acs.chemmater.4c01936](https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.4c01936)
- **Location:** `examples/LSCF_Poisoning_CaseStudy/`
- **Key Outcome:** Induced physical laws linking vacancy density and metal d-band center to poison binding strength.

### Sr Surface Segregation in LSF
A 50-iteration thermodynamic study of Sr enrichment in $La_{0.6}Sr_{0.4}FeO_3$ as a function of Temperature and Oxygen Pressure.
- **Location:** `examples/LSF_Segregation_CaseStudy/`
- **Key Outcome:** Mapping the $(T, P_{O2})$ drivers for segregation and the corresponding electronic d-band center shifts.

### B-site Doping Effects on SrTiO3
Autonomous screening of transition metal dopants (Mn, Fe, Co) on the $SrTiO_3$ (001) surface to activate oxygen adsorption.
- **Location:** `examples/SrTiO3_Doping_CaseStudy/`
- **Key Outcome:** Identified Mn as a high-performance dopant for enhancing oxygen affinity through localized electronic states.


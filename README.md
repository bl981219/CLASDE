# CLASDE v1: Formalized Closed-Loop Surface Optimization Engine

**CLASDE** (Closed-Loop Atomistic Surface Design Engine) is a mathematically defined constrained optimization framework. It treats the discovery of stable or high-performing surface configurations as a sequential decision process, orchestrated by a suite of specialized agents.

---

## 🎓 The Lab Metaphor: Roles & Responsibilities

To understand how CLASDE operates, imagine a high-performance computational chemistry research group. Each software component maps to a specific role in the lab.

### 🏛️ The Research Staff (Agents)

| Role | Agent | Metaphor | Responsibility |
| :--- | :--- | :--- | :--- |
| **Agent 1** | **Research Governor** | **The PI (Professor)** | Sets the high-level objective (Reward Function), defines the budget (Max Evaluations), and enforces constraints (Chemical Space). |
| **Agent 2** | **Optimization Strategist** | **The Senior Postdoc** | Operates the **Surrogate Model**. Uses Bayesian statistics and **Acquisition Functions** to decide which experiment is mathematically most likely to yield progress. |
| **Agent 3** | **Structure Builder** | **The PhD Student** | Takes abstract configuration descriptors and builds the actual 3D atomic structures using ASE and Pymatgen. |
| **Agent 4** | **Compute Manager** | **The Lab Technician** | Handles the "hardware" interface. Submits Slurm jobs to the HPC cluster, monitors progress, and manages input/output files. |
| **Agent 5** | **Evaluator** | **The Data Analyst** | Parses raw DFT outputs, extracts physical observables (Energy, Bader charges), and calculates the final "Score" (Reward). |
| **Agent 6** | **Memory Graph** | **The Lab Notebook** | A digital archive that records every state visited, every mutation performed, and every result ever computed. |

### 🔬 The Laboratory Infrastructure (Core Modules)

*   **`core/state.py` (The Sample):** The formal digital twin of a surface. It includes bulk composition, Miller indices, termination, and defects.
*   **`core/action.py` (The Protocol):** A set of discrete, enumerated mutation operators (e.g., "Add Vacancy", "Change Coverage"). No free-form guessing.
*   **`core/transition.py` (The Procedure):** The deterministic logic that transforms one Sample into another based on a Protocol.
*   **`core/surrogate.py` (The Intuition):** A Gaussian Process model that learns from past data to predict the outcome of future, unrun experiments.
*   **`core/reward.py` (The Metric):** Mathematical functions that turn physical data (eV) into a scalar "fitness" value.

---

## 🚀 Getting Started

### 📦 Installation

CLASDE requires a modern Python environment with atomistic simulation tools.

```bash
# Clone the repository
git clone <repo-url>
cd clasde_bill

# Install dependencies
pip install .
```

### 🔁 Running the Loop

The optimization loop is controlled by Agent 1's configuration. To start a discovery run:

```bash
python loop.py
```

---

## 🧬 Optimization Workflow

CLASDE follows a rigorous five-step cycle:

1.  **Strategic Decision:** The **Strategist** (Postdoc) looks at the **Memory** and updates the **Surrogate Model**. It selects the next best mutation using an Acquisition Function (e.g., Expected Improvement).
2.  **State Transition:** The **Governor** (PI) approves the move, and the **Transition Engine** generates the new configuration descriptor.
3.  **Physical Construction:** The **Builder** (PhD Student) converts the descriptor into a physical Slab with adsorbates.
4.  **HPC Execution:** The **Compute Manager** (Technician) ships the job to the cluster for DFT calculation.
5.  **Evaluation:** The **Evaluator** extracts the results, calculates the **Reward**, and saves it to **Memory**.

---

## 🛠️ Configuration

The `ResearchGovernor` is configured via a dictionary (or YAML) in `loop.py`:

```python
config = {
    "objective": {
        "type": "adsorption_tuning", 
        "target_e_ads": -1.5 # Target -1.5 eV
    },
    "budget": {
        "max_evaluations": 50
    },
    "acquisition": {
        "acquisition_type": "EI" # Expected Improvement
    }
}
```

---

## ⚠️ Safety & Constraints

*   **No Free-Text:** Agents cannot "hallucinate" new atoms. Every change must be an enumerated `MutationAction`.
*   **Budget Focused:** The loop terminates automatically when the PI's budget is exhausted.
*   **Serializable:** Every surface state is hashed (SHA-256) to ensure data integrity and prevent redundant calculations.

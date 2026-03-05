# CLASDE: Formalized Closed-Loop Surface Optimization Engine

**CLASDE** (Closed-Loop Atomistic Surface Design Engine) is a mathematically defined constrained optimization framework for the discovery of stable and high-performing surface configurations.

---

## 🚀 Key Features
- **Persistent Memory:** Support for session resumption (`results/clasde_memory.json`). Never lose progress.
- **VASP Integration:** High-fidelity DFT readiness with automated input generation and OUTCAR parsing.
- **Dynamic Exploration:** Agents now intelligently propose mutations based on the current surface stoichiometry.
- **Advanced Visualization:** Automated generation of optimization progress and exploration graph plots.
- **HPC Ready:** Multi-mode compute manager (Local EMT for testing, SLURM for production).

---

## 🎓 The Lab Metaphor: Roles & Responsibilities

| Role | Agent | Metaphor | Responsibility |
| :--- | :--- | :--- | :--- |
| **Agent 1** | **Research Governor** | **The PI (Professor)** | Sets the high-level objective, defines the budget, and enforces constraints. |
| **Agent 2** | **Optimization Strategist** | **The Senior Postdoc** | Operates the **Surrogate Model** (GPR) and selects the next experiment via Acquisition Functions (EI/UCB). |
| **Agent 3** | **Structure Builder** | **The PhD Student** | Builds 3D atomic structures using ASE/Pymatgen. |
| **Agent 4** | **Compute Manager** | **The Lab Technician** | Handles the HPC interface (Slurm) and input/output management. |
| **Agent 5** | **Evaluator** | **The Data Analyst** | Parses raw DFT outputs and calculates scalar rewards. |
| **Agent 6** | **Memory Graph** | **The Lab Notebook** | A digital archive of states, transitions, and observables. |

---

## 📦 Installation

```bash
pip install .
```

## 🔁 Usage

### Start Optimization
```bash
python loop.py
```

### Visualize Results
```bash
python visualize_loop.py
```
This generates `results/clasde_summary.png` containing the reward progress and the exploration graph.

---

## 🧬 Optimization Workflow
1. **Strategic Decision:** Strategist updates the GPR and selects the next mutation.
2. **State Transition:** Transition Engine generates the new configuration descriptor.
3. **Physical Construction:** Builder converts descriptor to a physical slab.
4. **Execution:** Compute Manager runs the calculation (Local or HPC).
5. **Evaluation:** Evaluator extracts physical data and updates Memory.

---

## ⚠️ Safety & Constraints
- **Strict Operations:** All mutations are from enumerated operators.
- **Reproducibility:** Every state is uniquely hashed (SHA-256).
- **Persistence:** Memory is serialized to JSON after every successful iteration.

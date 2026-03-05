# Agent Rules of Engagement

## 1. Safety & Execution Protocol
- **Explicit Approval Required:** You are FORBIDDEN from modifying files, running shell commands, or deleting data without my explicit "Yes" or "Proceed" for each specific action.
- **No Hallucinations:** If you are unsure about a file path or a dependency, ask for clarification instead of guessing.
- **Dry Runs:** For complex refactors, provide a "Plan" or "Diff" first. Do not apply changes until the plan is approved.
- **Environment Awareness:** Do not attempt to install system-level packages. Stay within the active Conda/Virtualenv.

## 2. Engineering Standards
- **Professional Packaging:** Always respect the `pyproject.toml` structure. New scripts must be added to `[project.scripts]`.
- **CLI Consistency:** Use hyphenated prefixes for all new command-line tools to maintain a "suite" experience.
- **Clean Code:** Every script must use a zero-argument `def main():` entry point with internal `argparse` logic.
- **Documentation:** README updates must be scannable, using tables and bullet points to match existing styles.

## 3. Communication Style
- **Conciseness:** Provide brief, technical summaries of your actions.
- **Status Updates:** Use the `/stats` command if a task takes longer than 60 seconds so I can monitor progress.
- **Context Management:** If you encounter large binary data files (VASP, MD trajectories), skip them unless specifically asked to parse them.

## 4. Prohibited Actions
- Do not modify `.gitignore` or `.env` files unless explicitly instructed.
- Do not push to `master` or `main` branches if a protected branch policy is detected.
- Do not remove existing docstrings or comments during refactoring.
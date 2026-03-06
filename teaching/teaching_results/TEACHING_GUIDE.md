# CLASDE Teaching Guide: Understanding the Research Log

This directory contains the output of a minimal 2-iteration research campaign. Use these files to teach users how CLASDE documents its discovery process.

## 1. The Research Log (`teaching_research_log.md`)
This file is the human-readable 'Master Record' of the campaign. Key sections include:
- **Original User Intent:** Shows the natural language prompt that started the research.
- **Exploration Phase:** A tabular record of every experiment, the action taken, and the physical reward observed.
- **Scientific Reasoning Phase:** Contains theories discovered by the PI (Principal Investigator) Agent.

## 2. The Memory Graph (`teaching_memory.json`)
This is the machine-readable 'Digital Lab Notebook'. It stores:
- Canonical descriptors of every surface configuration visited.
- The exact atomic structures used in calculations.
- Statistical patterns used for training future surrogate models.

## 3. How to use this for training
1. Open `teaching_research_log.md` in a Markdown viewer.
2. Trace how the system interpreted the prompt into a mathematical objective.
3. Observe how 'Vacancy_Density' or 'Coverage' were identified as driving factors in the theory section.

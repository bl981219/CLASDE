# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-04

### Added
- **LLM-Driven Reasoning**: Wired `PostdocAgent` to use Gemini API for idea critique, hypothesis formulation, and result analysis.
- **Hierarchical Discovery Loop**: Implemented the strict PI -> Postdoc -> Technician workflow in `CampaignManager`.
- **Epistemic Cycle**: Added iterative revision loop between PI and Postdoc for rejected ideas.
- **Semantic Memory Retrieval**: Implemented vector similarity search in `StorageRegistry` using `EmbeddingIndex`.
- **System Robustness**: Added consecutive error counter and tenacity retries for LLM calls.
- **Scientific Validation**: Re-implemented `TheoryBuilder` with actual linear regression for scaling relations and grouping for termination bias.
- **Testing Infrastructure**: Added `tests/conftest.py` and `pytest.ini` for better CI/CD compatibility.

### Changed
- **Statistically Sound GP**: Switched to anisotropic ARD kernels and implemented PCA dimensionality reduction for the surrogate model.
- **Formal Interface**: Enforced strict `Hypothesis` and `KnowledgeTrace` objects throughout the pipeline.
- **Dependency Pinning**: Pinned all scientific dependencies in `pyproject.toml` for exact reproducibility.

### Fixed
- Fixed `AttributeError` in `ScientificDiscoveryAcquisition` acquisition function.
- Fixed `.gitignore` regression that excluded test files.
- Removed runtime artifacts from git history.
- Resolved inline import issues and bare except blocks.

## [0.2.0] - 2026-04-04
- Initial refactor to hierarchical agent structure.
- Introduction of PostdocAgent role.
- Implementation of checkpointing and telemetry.

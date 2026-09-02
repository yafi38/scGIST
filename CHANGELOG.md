# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-09-02

### Changed

- **Breaking:** renamed the importable package from `scGIST` to lowercase `scgist`
  (`from scgist import scGIST`); the `scGIST` class name itself is unchanged.
- Migrated packaging to `pyproject.toml`, managed with [uv](https://docs.astral.sh/uv/);
  the version now comes from git tags via `hatch-vcs` instead of being hardcoded.
- Restructured into a `src/scgist/` layout; split `utility.py` into `evaluation.py`,
  `priority.py`, and `plotting.py`, and renamed `scGIST.py` → `model.py` and
  `customLayers.py` → `layers.py`.
- Bumped all dependencies to modern versions (TensorFlow, NumPy, pandas,
  scikit-learn). Dropped the `scanpy` runtime dependency in favor of the much
  lighter `anndata`; `scanpy` is now only needed for `benchmarks/`.
- Now supports Python 3.10–3.13 (previously pinned to Python 3.7).
- Moved example notebooks into `examples/`.

### Added

- Type hints across the public API, plus a `py.typed` marker (PEP 561).
- `mypy` and `ruff` configuration, checked in CI.
- A pytest suite covering `FeatureRegularizer`, `OneToOneLayer`,
  `get_priority_score_list`, `scGIST`, and `test_classifier`.
- GitHub Actions CI workflow: lint, type-check, and test on Python 3.10–3.13.
- Published to PyPI as `scgist`, with a GitHub Actions release workflow that
  publishes via PyPI trusted publishing (OIDC) on each GitHub Release.

### Fixed

- `scGIST.train_model`, `scGIST.get_markers_names`, and `test_classifier` now
  raise `ValueError` on missing required arguments instead of silently
  printing a message and returning `None`.
- `plot_marker_weights` no longer passes an invalid argument shape to
  matplotlib's `xlim`.
- Removed a phantom `sklearn~=0.0` dependency that broke installs outright.

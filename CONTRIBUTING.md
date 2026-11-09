# Contributing to ConfiDock

Contributions are welcome. This document explains how to report issues and propose changes.

## Reporting issues

Please open a GitHub issue that includes:

- A clear description of the problem or request.
- Steps to reproduce (for bugs), including the ConfiDock version, Python version, platform,
  and — for anything touching inference — which co-folding backend and GPU you used.
- The expected versus actual behaviour.

Do **not** include any confidential, personal, or patient data in issues, pull requests, or
test fixtures. The same applies to sequences or structures you are not free to publish.

## Proposing changes

1. Fork the repository and create a feature branch off the default branch.
2. Set up the development environment and install the pre-commit hooks:
   ```bash
   make init
   ```
3. Make your change, keeping it focused and well described.
4. Add or update tests under `tests/` so the behaviour is covered. Tests must run on CPU
   with no network access and no model weights; mark anything that needs a real
   checkpoint with `@pytest.mark.slow`.
5. Run the checks locally before opening a pull request:
   ```bash
   poetry run pre-commit run --all-files
   poetry run pytest -m "not slow"
   ```
6. Open a pull request describing what changed and why. Link any related issue.

## Coding conventions

- Formatting is enforced with **black** (line length 120) and linting with **ruff**, both run
  automatically by the pre-commit hooks.
- Docstrings follow the **NumPy** convention.
- Paths that depend on where the code is deployed belong in `confidock/paths.py` behind a
  `CONFIDOCK_*` environment variable, never hardcoded in a module or a script default.
- Keep public API changes backward compatible where possible; call out any breaking change in
  your pull request description.

## Versioning

ConfiDock follows [semantic versioning](https://semver.org/).

## Contact

For questions that are not suitable for a public issue, contact the maintainer listed in
[AUTHORS.md](AUTHORS.md).

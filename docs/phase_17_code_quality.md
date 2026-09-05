# Phase 17 — Code quality review

## Review results

| Area | Result |
| --- | --- |
| Repeated paths, seed, and dataset columns | Centralized in `src/config.py`. |
| Data leakage | Learned preprocessing remains inside fitted Scikit-learn pipelines; tuning uses training-only CV. |
| Duplicated cleaning/feature code | Reusable logic is in `src/data/` and `src/features/`; notebooks import it. |
| Model artifact | A complete pipeline is serialized, rather than a bare estimator. |
| API behavior | Validation, model-load, and unexpected-error paths are handled without stack-trace exposure. |
| Test coverage | API, preprocessing, and feature-engineering behavior is covered by seven focused tests. |

## Dependency and Python review

The old `requirements.txt` was a 132-line environment snapshot containing indirect Jupyter and platform-specific packages. It has been replaced with the small, pinned set of direct application, exploration, and test dependencies. Pip now resolves appropriate transitive packages for the active platform instead of attempting to reproduce a Mac-specific development environment in Docker.

`pyproject.toml` declares Python **3.10–3.12**. This matches the Docker Python 3.11 image and prevents attempting a fresh install with Python 3.9, which caused the earlier `pydantic_core` resolution failure.

## Remaining intentional trade-offs

- The API requires all raw Ames fields because the deployed pipeline was trained on that schema. A smaller product-facing request schema would require retraining a deliberately smaller model.
- Expensive homes have materially higher error; the API should be positioned as an estimate, not an appraisal.
- The Docker daemon was not running during validation, so the Dockerfile has been reviewed but still needs a local `docker build` once Docker Desktop is active.

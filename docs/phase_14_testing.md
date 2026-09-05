# Phase 14 — Testing

The test suite is deliberately small and behavior-focused.

| Test module | Coverage |
| --- | --- |
| `tests/test_api.py` | Health endpoint, successful prediction, and invalid request validation. |
| `tests/test_preprocessing.py` | Deterministic cleaning, quality reporting, and raw-layer write protection. |
| `tests/test_feature_engineering.py` | Exact values for every engineered feature and non-mutation of input data. |

Run the suite from the repository root:

```bash
python -m pytest
```

The API test uses the serialized model artifact under `models/`. Train/serialize the model (Phase 11) before running it in a fresh clone or CI environment.

# Phase 16 — Project configuration

`src/config.py` is the central source of truth for environment-independent project values:

| Value | Purpose |
| --- | --- |
| `PROJECT_ROOT` | Absolute project root derived from this source file. |
| `RAW_TRAIN_PATH`, `PROCESSED_TRAIN_PATH` | Standard data-layer locations. |
| `MODEL_PATH` | Serialized complete pipeline location. |
| `TARGET_COLUMN`, `IDENTIFIER_COLUMN` | Dataset column names shared by cleaning and training. |
| `RANDOM_STATE`, `TEST_SIZE` | Reproducible experiment settings. |

Data preprocessing, model training, model loading, and FastAPI now import these values instead of defining duplicate paths or seeds. This keeps future path or experiment-setting changes local and auditable.

# Phase 15 — Docker

## Build and run

The trained artifact must exist locally before building the image:

```bash
docker build -t house-price-api .
docker run --rm -p 8000:8000 house-price-api
```

Then visit `http://localhost:8000/docs`.

## Dockerfile decisions

| Instruction | Purpose |
| --- | --- |
| `FROM python:3.11-slim` | Uses a small Linux runtime with a supported modern Python version. |
| `ENV` | Disables bytecode files, makes logs immediate, and prevents pip cache bloat. |
| `WORKDIR /app` | Provides a stable application directory. |
| `COPY requirements.txt` then `RUN pip install` | Lets Docker cache dependency installation when application code changes. |
| `useradd` and `USER appuser` | Runs the service without root privileges. |
| `COPY src`, `app`, and pipeline artifact | Includes only code and the model required to serve predictions. |
| `EXPOSE 8000` | Documents the Uvicorn service port. |
| `CMD` | Starts the API bound to all container interfaces. |

`.dockerignore` keeps raw/processed data, notebooks, tests, local virtual environments, and Git metadata out of the image. The serialized `house_price_pipeline.joblib` is explicitly retained because the API needs it at startup.

# Phase 13 — API error handling

## Client-facing responses

| Situation | Status | Response behavior |
| --- | ---: | --- |
| Invalid value, missing required field, or unexpected field | 422 | Clear generic message plus the invalid field and validation message. Request values are not echoed back. |
| Model failed to load at startup | 503 | `"Prediction model is unavailable."` |
| Unexpected prediction failure | 500 | `"An unexpected server error occurred."` |

## Server-side diagnostics

Model-loading and unexpected errors are logged with their full exception context. The client receives only a safe, actionable response; internal paths, implementation details, and stack traces are never returned in JSON.

`GET /health` exposes only whether the model loaded (`ok` or `degraded`), not the internal load failure reason.

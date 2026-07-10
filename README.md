# Emerging Risk Monitor

AI service for classifying emerging-risk text, answering risk questions with retrieval support, and generating asynchronous risk reports. The project exposes a Flask API backed by Groq-hosted LLM calls, ChromaDB retrieval, optional Redis caching, and structured response metadata for latency, model, cache, and fallback visibility.

## What This Project Demonstrates

- Practical backend AI API design with predictable request and response shapes
- Retrieval-aware question answering over a local ChromaDB collection
- Risk categorisation and report-generation workflows
- Runtime metadata for monitoring latency, cache behavior, fallback paths, and model usage
- A lightweight service layout that can be containerized and deployed independently

## Features

- `POST /categorise` classifies risk text and returns category, confidence, and reasoning.
- `POST /query` answers questions using the query service and source retrieval.
- `POST /generate-report` starts an asynchronous report-generation job.
- `GET /generate-report/<job_id>` checks report-job status and result payloads.
- `GET /health` reports service status, model name, cache statistics, runtime metrics, and indexed document count.
- Optional fresh-query mode through `?fresh=true`, `X-Fresh-Request: true`, or JSON `{ "fresh": true }`.

## Architecture

```text
Client
  |
  v
Flask API (`ai-service/app.py`)
  |
  +-- Categorisation service
  +-- Query service
  |     +-- ChromaDB vector store
  |     +-- Optional Redis/cache layer
  |     +-- Groq LLM client
  +-- Report job service
  +-- Runtime metrics and response metadata
```

## Tech Stack

| Area | Tools |
| --- | --- |
| API | Flask |
| LLM provider | Groq SDK |
| Retrieval | ChromaDB, sentence-transformers |
| Cache / jobs | Redis-compatible services |
| Runtime | Python, Gunicorn, Docker |
| Configuration | `.env` / environment variables |

## Repository Structure

```text
.
├── .env.example
├── README.md
└── ai-service/
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    ├── prompts/
    ├── routes/
    ├── services/
    │   ├── categoriser.py
    │   ├── chroma_store.py
    │   ├── groq_client.py
    │   ├── query_service.py
    │   ├── report_job.py
    │   └── runtime_metrics.py
    └── verify_redis_cache.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- A Groq API key
- Redis, if you want cache/job behavior that depends on Redis

### Installation

```bash
git clone https://github.com/Lakshmish18/emerging-risk-monitor.git
cd emerging-risk-monitor/ai-service
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configuration

Create a `.env` file from the sample at the repository root or export the same variables in your shell.

Common variables:

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | API key used by the Groq client |
| `REDIS_URL` | Redis connection string, when cache/report storage is enabled |
| `CHROMA_DIR` | Local ChromaDB persistence path, if supported by the service config |

## Running Locally

```bash
cd ai-service
flask --app app run --debug --port 5000
```

Production-style run:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

## API Examples

Categorise text:

```bash
curl -X POST http://localhost:5000/categorise \
  -H "Content-Type: application/json" \
  -d '{"text":"Supplier concentration and geopolitical disruption may affect delivery timelines."}'
```

Ask a retrieval-backed question:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Which operational risks should be prioritised this week?"}'
```

Start a report job:

```bash
curl -X POST http://localhost:5000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"brief":"Summarise the top emerging supplier and market risks."}'
```

## Testing and Verification

The repository includes verification scripts for ChromaDB, Groq connectivity, prompt quality, Redis cache behavior, and demo QA flows.

```bash
cd ai-service
python test_groq.py
python test_chromadb.py
python verify_redis_cache.py
python prompt_eval.py
```

Run only the checks that match your configured environment. Scripts that call external services require valid credentials.

## Production Readiness Notes

Recommended next hardening steps:

- Add request authentication before exposing the API publicly.
- Add structured JSON logging and request IDs across service boundaries.
- Add rate limiting for LLM-backed endpoints.
- Add unit tests for request validation, fallback responses, and cache behavior.
- Add CI once tests can run without external credentials.
- Document the exact `.env.example` variables and default values used by each service module.

## Security

- Do not commit `.env` files or API keys.
- Keep provider keys in a secret manager for deployments.
- Treat generated reports and retrieved sources as potentially sensitive operational data.
- Validate webhook URLs before enabling outbound callbacks in production.

## License

No license file is currently included. Add a license before accepting outside contributions or reuse.

## Contact

Lakshmish M Devadiga

- GitHub: [Lakshmish18](https://github.com/Lakshmish18)
- LinkedIn: [lakshmish-m-devadiga](https://www.linkedin.com/in/lakshmish-m-devadiga)
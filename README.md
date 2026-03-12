# Bayesian Probabilistic LangChain API

This repository packages the Bayesian Probabilistic LangChain runtime as a portable FastAPI service with tests, CI, container support, and reproducible demo scripts.

## What is in this repository

There are two layers here.

The preserved research runtime lives under `src/b_langchain/` and `src/p_langchain/`. The production-facing API package lives under `src/bayesian_prob_langchain_api/`. This layer adds:

- a clean FastAPI application factory
- versioned endpoints under `/api/v1`
- request and response schemas
- a service container and backend abstraction
- a lightweight built-in retriever for smoke testing and API demos
- a test suite that runs without external LLM credentials
- CI and Docker support
- benchmark and demo scripts for the accompanying software paper

## Repository layout

```text
src/
  b_langchain/                     preserved Bayesian LangChain runtime
  p_langchain/                     preserved probabilistic support code
  bayesian_prob_langchain_api/
    api/
      app.py                       FastAPI app factory and module-level app
      deps.py                      dependency wiring
      routes/
        health.py                  health and info endpoints
        tools.py                   tool listing endpoint
        agents.py                  orchestration endpoint
        rag.py                     retrieval endpoint
    services/
      container.py                 tool and backend registry
      orchestrator.py              agent execution wrapper
      rag.py                       retrieval wrapper
      tools.py                     local tools used by the API
    config.py                      simple runtime settings
    schemas.py                     pydantic models
tests/
  test_api.py
  test_agents.py
  test_rag.py
  test_validation.py
benchmarks/
  primary_benchmark.py             non-LLM repeated robustness benchmark
examples/
  normal_rag_demo.py               end-to-end open-source RAG demo
  pathological_rag_demo.py         greedy vs Bayesian pathological RAG comparison
  legacy_demos/                    preserved original demo scripts
.github/workflows/ci.yml           GitHub Actions workflow
Dockerfile                         container entry point
pyproject.toml                     install and dependency metadata
README.md                          reviewer-facing documentation
LICENSE                            MIT license
```

## API endpoints

The public REST surface is intentionally small.

- `GET /api/v1/health`
- `GET /api/v1/info`
- `GET /api/v1/tools`
- `POST /api/v1/agents/run`
- `POST /api/v1/rag/query`

### Health check

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Expected response shape:

```json
{
  "ok": true,
  "service": "Bayesian Probabilistic LangChain API",
  "version": "0.2.0",
  "available_backends": ["local"]
}
```

### Service metadata

```bash
curl http://127.0.0.1:8000/api/v1/info
```

### List available tools

```bash
curl http://127.0.0.1:8000/api/v1/tools
```

### Run the Bayesian orchestrator

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "calculate 2 + 3",
    "backend": "local",
    "init_state": {}
  }'
```

Typical response fields include the best action, tool payload, observation, posterior probabilities, and execution metadata returned by the runtime.

### Query the built-in retriever

```bash
curl -X POST http://127.0.0.1:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "backend agnostic runtime",
    "top_k": 2
  }'
```

The built-in retrieval endpoint is intentionally lightweight. It is meant for smoke testing, examples, and API validation. It is not presented as a production enterprise RAG stack.

## Installation

### Local development install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

### Install with optional Gemini backend

```bash
pip install -e .[dev,gemini]
```

### Install with optional open-source RAG demo dependencies

```bash
pip install -e .[dev,rag]
```

## Running the API

```bash
uvicorn bayesian_prob_langchain_api.api.app:app --host 0.0.0.0 --port 8000
```

The factory form also works:

```bash
uvicorn bayesian_prob_langchain_api.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

## Running tests

```bash
pytest -q
```

The tests are designed to pass without external API keys.

## Running in Docker

```bash
docker build -t bayesian-prob-langchain-api .
docker run --rm -p 8000:8000 bayesian-prob-langchain-api
```

## Backends

The service always exposes the `local` backend. It only advertises `gemini` when the optional Gemini dependency is installed and credentials are available at runtime.

### `local`
This is the default portable backend. It uses the local heuristic LLM wrapper already present in the runtime, and it is the backend used by the automated tests.

### `gemini`
This option is available only when the optional dependency is installed and the original Gemini wrapper is configured in the environment. Use this backend only when you have the correct credentials in place.

## Benchmark and demo scripts

### 1. Primary benchmark

This is the main repeated non-LLM software benchmark for the paper narrative.

```bash
python benchmarks/primary_benchmark.py
```

It runs 100 trials across four scenario families:

- stale versus verified sources
- session learning after early mistakes
- ambiguous location/entity resolution
- noisy web evidence versus official database evidence

The script writes `primary_benchmark_results.json`.

### 2. Normal RAG demo

```bash
pip install -e .[rag]
python examples/normal_rag_demo.py
```

This runs a small end-to-end open-source RAG example using a sentence-transformer retriever and a compact sequence-to-sequence generator.

### 3. Pathological RAG comparison

```bash
pip install -e .[rag]
python examples/pathological_rag_demo.py
```

This is the important RAG comparison for the paper. It compares a greedy single-path retrieval strategy against a Bayesian multi-hypothesis strategy in the presence of stale but lexically tempting evidence.

## Reviewer notes

This repository is intentionally positioned as research software.

The goal is not to hide the research provenance of the runtime. The goal is to make the runtime installable, testable, callable through a stable API, and easy to evaluate.

A few design choices follow from that.

- The runtime in `src/b_langchain/` is preserved rather than heavily rewritten.
- The API layer is thin and explicit.
- The built-in retriever is simple by design so the package remains portable.
- The more expensive transformer-based RAG examples live in `examples/`, not in the core API path.
- The non-LLM robustness benchmark is separated into `benchmarks/` so reviewers can reproduce the headline table without needing model downloads.

## Suggested GitHub workflow

1. Unzip this package into a fresh branch.
2. Review the new `src/`, `tests/`, `benchmarks/`, and `examples/` folders.
3. Run `pip install -e .[dev]`.
4. Run `pytest -q`.
5. Launch the API with `uvicorn`.
6. Optionally run the benchmark and RAG demos.
7. Push to GitHub and enable Actions so CI runs on every commit.

## Known limitations

- The built-in retriever is lexical and lightweight. It is meant for portability, not state-of-the-art retrieval quality.
- The transformer-based RAG demos are examples, not optimized production deployments.
- The `gemini` backend depends on external credentials and should be treated as optional.
- The benchmark scripts are software evaluation harnesses, not claims about general LLM reasoning performance.

## Why this repository is now portable

This package can be installed with `pip`, tested with `pytest`, served with `uvicorn`, containerized with Docker, and versioned through standard Python packaging. That is the minimum bar a reviewer expects when assessing reusable research software.

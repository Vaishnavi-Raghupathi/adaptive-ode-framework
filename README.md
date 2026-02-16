# ODE Framework

A production-ready framework for solving ordinary differential equations (ODEs) with classical numerical methods, a REST API for solver jobs, and persistent job tracking.

## Features

- **Classical ODE solvers** — RK45, RK4 and other scipy-based methods with configurable polynomial-order dynamics
- **REST API for solver jobs** — Submit solve requests, poll job status, and retrieve metrics
- **PostgreSQL job tracking** — Persistent storage of job metadata, parameters, and accuracy metrics
- **Docker development environment** — API, PostgreSQL, and Redis services via Docker Compose
- **Comprehensive test suite** — Pytest tests for solvers and API, with SQLite-backed test database

## Quick Start

### Prerequisites

- **Docker** and Docker Compose (for running the full stack)
- **Python 3.9+** (for local development and tests)

### Clone and install

```bash
git clone <repository-url>
cd adaptive-ode-framework-1
pip install -r requirements-dev.txt
```

### Start services with Docker

```bash
cd docker && docker-compose up --build
```

The API will be available at `http://localhost:8000`. PostgreSQL is on port 5432 and Redis on 6379.

### Run tests

```bash
pytest
```

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{"status": "healthy"}
```

### Submit a solve job

```bash
curl -X POST http://localhost:8000/api/v1/solve \
  -H "Content-Type: application/json" \
  -d '{
    "t_data": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    "x_data": [[1.0], [0.819], [0.670], [0.549], [0.449], [0.368], [0.301], [0.247], [0.202], [0.165], [0.135]],
    "solver_type": "RK45",
    "order": 2
  }'
```

Example response:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "message": "Job completed successfully."
}
```

### Check job status

Replace `{job_id}` with the `job_id` from the solve response:

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

Example response:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "solver_type": "RK45",
  "metrics": {"mse": 0.0012, "rmse": 0.035, "r2": 0.998},
  "created_at": "2025-02-16T12:00:00",
  "completed_at": "2025-02-16T12:00:05"
}
```

## Development

### Pre-commit hooks

Install hooks to run black, flake8, and mypy before each commit:

```bash
pre-commit install
```

### Formatting

```bash
black .
```

### Linting

```bash
flake8 ode_framework/
```

### Type checking

```bash
mypy ode_framework/
```

## Project structure

```
adaptive-ode-framework-1/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── docker/
│   ├── Dockerfile              # API image (Python 3.9, uvicorn)
│   └── docker-compose.yml      # api, postgres, redis
├── ode_framework/
│   ├── api/
│   │   ├── main.py             # FastAPI app, /health, /api/v1/solve, /api/v1/jobs/{job_id}
│   │   └── schemas.py          # Pydantic request/response models
│   ├── models/
│   │   └── database.py         # SQLAlchemy engine, SolverJob, init_db
│   ├── solvers/
│   │   ├── base.py             # BaseSolver abstract class
│   │   └── classical.py        # ClassicalSolver (scipy)
│   ├── tests/
│   │   ├── conftest.py         # pytest fixtures (test DB, client)
│   │   └── test_classical_solvers.py
│   └── utils/
│       └── config.py           # pydantic-settings (database_url, redis_url)
├── .dockerignore
├── .pre-commit-config.yaml
├── pyproject.toml              # black, mypy, pytest config
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

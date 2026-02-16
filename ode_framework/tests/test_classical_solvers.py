"""Tests for ClassicalSolver and solve API endpoints."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from ode_framework.solvers.classical import ClassicalSolver


def _exponential_decay_data(
    n_points: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate t in [0, 2] and x = exp(-t) as (n_points,) and (n_points, 1)."""
    t_data = np.linspace(0.0, 2.0, n_points, dtype=np.float64)
    x_data = np.exp(-t_data).reshape(-1, 1).astype(np.float64)
    return t_data, x_data


def test_classical_solver_initialization() -> None:
    """ClassicalSolver with RK45 and order=2 has correct name and fitted flag."""
    solver = ClassicalSolver(method="RK45", order=2)
    assert solver.name == "Classical_RK45"
    assert solver.fitted is False


def test_classical_solver_fit() -> None:
    """Solver fits exponential decay data and sets fitted and theta."""
    t_data, x_data = _exponential_decay_data()
    solver = ClassicalSolver(method="RK45", order=2)
    solver.fit(t_data, x_data)
    assert solver.fitted is True
    assert solver.theta is not None


def test_classical_solver_predict() -> None:
    """Predict returns array with correct shape (n_times, n_states)."""
    t_data, x_data = _exponential_decay_data()
    solver = ClassicalSolver(method="RK45", order=2)
    solver.fit(t_data, x_data)
    t_eval = np.linspace(0.0, 1.5, 20, dtype=np.float64)
    x0 = x_data[0, :].copy()
    out = solver.predict(t_eval, x0)
    assert out.shape == (20, 1)


def test_classical_solver_metrics() -> None:
    """get_metrics returns mse, rmse, r2 and MSE is non-negative."""
    t_data, x_data = _exponential_decay_data()
    solver = ClassicalSolver(method="RK45", order=2)
    solver.fit(t_data, x_data)
    metrics = solver.get_metrics(t_data, x_data)
    assert "mse" in metrics
    assert "r2" in metrics
    assert "rmse" in metrics
    assert metrics["mse"] >= 0


def test_predict_without_fit_raises_error() -> None:
    """Calling predict before fit raises RuntimeError."""
    solver = ClassicalSolver(method="RK45", order=2)
    t_eval = np.array([0.0, 1.0], dtype=np.float64)
    x0 = np.array([[1.0]], dtype=np.float64)
    with pytest.raises(RuntimeError):
        solver.predict(t_eval, x0)


def test_api_health_check(client: TestClient) -> None:
    """GET /health returns 200 and {'status': 'healthy'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_api_solve_endpoint(client: TestClient) -> None:
    """POST /api/v1/solve with exponential decay data returns 200 and completed job."""
    t_data, x_data = _exponential_decay_data(n_points=15)
    payload = {
        "t_data": t_data.tolist(),
        "x_data": x_data.tolist(),
        "solver_type": "RK45",
        "order": 2,
    }
    response = client.post("/api/v1/solve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "completed"

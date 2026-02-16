"""Classical ODE solver using scipy.integrate and scipy.optimize."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score

from ode_framework.solvers.base import BaseSolver

# Large penalty for integration failures in optimization
_FAILURE_PENALTY = 1e10


class ClassicalSolver(BaseSolver):
    """ODE solver that fits polynomial dynamics using scipy."""

    def __init__(
        self,
        method: str = "RK45",
        order: int = 2,
    ) -> None:
        super().__init__(name=f"Classical_{method}")
        self.method: str = method
        self.order: int = order
        self.theta: NDArray[np.float64] | None = None
        self._n_states: int | None = None

    @staticmethod
    def _polynomial_dynamics(
        t: float,
        x: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute dx/dt = sum(theta_i * x^i) for each state dimension.

        For multi-dimensional x of shape (n_states,), theta has shape
        (n_states, order+1) and dx[d] = sum_k theta[d, k] * x[d]**k.

        Args:
            t: Time (unused; for solve_ivp compatibility).
            x: State vector, shape (n_states,).
            theta: Parameters, shape (n_states, order+1).

        Returns:
            dx/dt, shape (n_states,).
        """
        n_states = x.shape[0]
        order_plus_one = theta.shape[1]
        dx = np.zeros_like(x, dtype=np.float64)
        for d in range(n_states):
            for k in range(order_plus_one):
                dx[d] += theta[d, k] * (x[d] ** k)
        return dx

    def _objective(
        self,
        theta_flat: NDArray[np.float64],
        t_data: NDArray[np.float64],
        x_data: NDArray[np.float64],
    ) -> float:
        """MSE between model predictions and data. Large penalty on integration failure."""
        n_states = x_data.shape[1]
        order_plus_one = self.order + 1
        theta = theta_flat.reshape(n_states, order_plus_one)

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return ClassicalSolver._polynomial_dynamics(t, y, theta)

        x0 = x_data[0, :].copy()
        t_span = (float(t_data[0]), float(t_data[-1]))
        try:
            result = solve_ivp(
                rhs,
                t_span,
                x0,
                method=self.method,
                t_eval=t_data,
                dense_output=False,
            )
        except Exception:
            return _FAILURE_PENALTY

        if not result.success:
            return _FAILURE_PENALTY

        # result.y has shape (n_states, n_times)
        x_pred = result.y.T  # (n_times, n_states)
        mse = float(mean_squared_error(x_data, x_pred))
        return mse

    def fit(
        self,
        t_data: NDArray[np.float64],
        x_data: NDArray[np.float64],
    ) -> "ClassicalSolver":
        """Fit polynomial dynamics to observed data via L-BFGS-B."""
        t_data = np.asarray(t_data, dtype=np.float64)
        x_data = np.asarray(x_data, dtype=np.float64)
        if t_data.ndim != 1:
            raise ValueError("t_data must be 1-dimensional")
        if x_data.ndim != 2:
            raise ValueError("x_data must be 2-dimensional (n_times, n_states)")
        if len(t_data) != x_data.shape[0]:
            raise ValueError("t_data length must match x_data number of rows")

        n_states = x_data.shape[1]
        order_plus_one = self.order + 1
        size = n_states * order_plus_one
        np.random.seed(None)
        theta0 = np.random.randn(size).astype(np.float64) * 0.1

        def objective_wrapper(theta_flat: NDArray[np.float64]) -> float:
            return self._objective(theta_flat, t_data, x_data)

        res = minimize(
            objective_wrapper,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.theta = res.x.reshape(n_states, order_plus_one)
        self._n_states = n_states
        self.fitted = True
        return self

    def predict(
        self,
        t_eval: NDArray[np.float64],
        x0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict state at t_eval using fitted dynamics."""
        if not self.fitted or self.theta is None:
            raise RuntimeError("Solver must be fitted before predict (call fit first)")
        t_eval = np.asarray(t_eval, dtype=np.float64)
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape[0] != self.theta.shape[0]:
            raise ValueError(
                f"x0 dimension {x0.shape[0]} does not match fitted n_states {self.theta.shape[0]}"
            )

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            return ClassicalSolver._polynomial_dynamics(t, y, self.theta)

        t_span = (float(t_eval[0]), float(t_eval[-1]))
        result = solve_ivp(
            rhs,
            t_span,
            x0,
            method=self.method,
            t_eval=t_eval,
            dense_output=False,
        )
        if not result.success:
            msg = getattr(result, "message", "unknown")
            raise RuntimeError(f"Integration failed: {msg}")
        return result.y.T.astype(np.float64)  # (n_times, n_states)

    def get_metrics(
        self,
        t_test: NDArray[np.float64],
        x_test: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute MSE, RMSE, and R² between predictions and test data."""
        x0 = x_test[0, :].copy()
        x_pred = self.predict(t_test, x0)
        mse = float(mean_squared_error(x_test, x_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(x_test, x_pred))
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
        }

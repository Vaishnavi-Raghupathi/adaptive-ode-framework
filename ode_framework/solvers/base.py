"""Abstract base class for ODE solvers."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseSolver(ABC):
    """Abstract base class for all ODE solvers."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.fitted: bool = False

    @abstractmethod
    def fit(
        self,
        t_data: NDArray[np.float64],
        x_data: NDArray[np.float64],
    ) -> "BaseSolver":
        """Fit the solver to observed data.

        Args:
            t_data: Time points of observations.
            x_data: State values at each time point.

        Returns:
            self for method chaining.
        """
        ...

    @abstractmethod
    def predict(
        self,
        t_eval: NDArray[np.float64],
        x0: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Generate predictions at new time points.

        Args:
            t_eval: Time points at which to evaluate the solution.
            x0: Initial state (e.g. at t_eval[0] or a known initial condition).

        Returns:
            Predicted state values at each time in t_eval.
        """
        ...

    @abstractmethod
    def get_metrics(
        self,
        t_test: NDArray[np.float64],
        x_test: NDArray[np.float64],
    ) -> dict[str, float]:
        """Calculate accuracy metrics against test data.

        Args:
            t_test: Test time points.
            x_test: Ground-truth state values at test times.

        Returns:
            Dictionary of metric names to float values.
        """
        ...

"""Pydantic schemas for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    """Request body for submitting an ODE solve job."""

    t_data: list[float] = Field(
        ...,
        description="Time points of observations.",
    )
    x_data: list[list[float]] = Field(
        ...,
        description="State observations at each time point (n_times x n_states).",
    )
    solver_type: str = Field(
        default="RK45",
        description="Integration method (e.g. RK45, BDF).",
    )
    order: int = Field(
        default=2,
        description="Polynomial order for fitted dynamics.",
        ge=1,
        le=10,
    )


class SolveResponse(BaseModel):
    """Response after submitting a solve job."""

    job_id: str = Field(
        ...,
        description="Unique identifier for the submitted job.",
    )
    status: str = Field(
        ...,
        description="Current job status (e.g. pending, running).",
    )
    message: str = Field(
        ...,
        description="Human-readable message about the submission.",
    )


class JobStatus(BaseModel):
    """Full status and result of a solve job."""

    job_id: str = Field(
        ...,
        description="Unique identifier for the job.",
    )
    status: str = Field(
        ...,
        description="Job status: pending, running, completed, or failed.",
    )
    solver_type: str | None = Field(
        default=None,
        description="Solver/integration method used for this job.",
    )
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Accuracy metrics (e.g. MSE, RMSE, R²) when completed.",
    )
    created_at: datetime = Field(
        ...,
        description="When the job was created.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When the job finished (if completed or failed).",
    )

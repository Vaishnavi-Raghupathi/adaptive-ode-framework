"""Main FastAPI application for the ODE Framework API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generator
import uuid

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from ode_framework.api.schemas import JobStatus, SolveRequest, SolveResponse
from ode_framework.models.database import SolverJob, init_db, SessionLocal
from ode_framework.solvers.classical import ClassicalSolver


app = FastAPI(
    title="ODE Framework API",
    version="0.1.0",
)


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and ensure it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    """Create database tables on application startup."""
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/solve", response_model=SolveResponse)
def solve(
    request: SolveRequest,
    db: Annotated[Session, Depends(get_db)],
) -> SolveResponse:
    """Submit an ODE solve job and run it synchronously."""
    job_id = str(uuid.uuid4())
    job = SolverJob(
        job_id=job_id,
        solver_type=request.solver_type,
        status="running",
        parameters={"order": request.order},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        t_data = np.array(request.t_data, dtype=np.float64)
        x_data = np.array(request.x_data, dtype=np.float64)

        solver = ClassicalSolver(method=request.solver_type, order=request.order)
        solver.fit(t_data, x_data)
        metrics = solver.get_metrics(t_data, x_data)

        job.status = "completed"
        job.metrics = metrics
        job.completed_at = datetime.utcnow()
        db.commit()

        return SolveResponse(
            job_id=job_id,
            status="completed",
            message="Job completed successfully.",
        )
    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> JobStatus:
    """Get the status and result of a solve job by job_id."""
    job = db.query(SolverJob).filter(SolverJob.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        solver_type=job.solver_type,
        metrics=job.metrics,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )

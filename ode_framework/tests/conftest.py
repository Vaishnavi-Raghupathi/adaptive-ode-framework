"""Pytest configuration and shared fixtures."""

from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ode_framework.api.main import app, get_db
from ode_framework.models.database import Base


TEST_DATABASE_URL = "sqlite:///./test.db"


@pytest.fixture(scope="function")
def test_db() -> Generator[Any, None, None]:
    """Provide a test database with overridden get_db dependency.

    Creates a SQLite engine and tables, overrides the app's get_db to use
    the test session, then drops all tables on cleanup.
    """
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestingSessionLocal
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db: Any) -> TestClient:
    """Return a TestClient for the FastAPI app using the test database."""
    return TestClient(app)

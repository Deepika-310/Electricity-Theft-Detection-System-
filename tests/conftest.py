import warnings

import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_detector
from backend.services import FraudDetector
from database.db import get_connection
from database.init_db import initialize_database
from model.train import train

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def detector(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("etd")
    db_path = str(tmp / "test.db")
    model_path = str(tmp / "model.pkl")
    initialize_database(db_path=db_path, seed=7, n_meters=5, days=20)
    train(db_path=db_path, model_path=model_path)
    return FraudDetector(connection_factory=lambda: get_connection(db_path), model_path=model_path)


@pytest.fixture()
def client(detector):
    app.dependency_overrides[get_detector] = lambda: detector
    yield TestClient(app)
    app.dependency_overrides.clear()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from faircredit import load_german_credit, train_primary_model


@pytest.fixture(scope="session")
def data():
    return load_german_credit()


@pytest.fixture(scope="session")
def training(data):
    return train_primary_model(data)

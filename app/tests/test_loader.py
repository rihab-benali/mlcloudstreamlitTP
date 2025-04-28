import os
import pytest
from app.model_loader import load_my_model

def test_model_file_exists():
    assert os.path.exists("app/model.h5")

def test_model_loads():
    model = load_my_model()
    assert model is not None

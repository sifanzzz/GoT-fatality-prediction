import pytest
from src.model_train import create_model
from src.data_pipeline import create_pipeline
import pandas as pd
import numpy as np
import os
import pickle


@pytest.fixture
def sample_data():
    data = {
        "title": ["Lord of the Crossing", "Ser", "Queen"],
        "male": [1, 0, 0],
        "culture": ["Rivermen", "", "Dornish"],
        "dateOfBirth": [208, 0, 82],
        "house": ["House Frey", "House Swyft", "House Santagar"],
        "book1": [1, 0, 0],
        "book2": [1, 1, 0],
        "book3": [1, 0, 0],
        "book4": [1, 0, 0],
        "book5": [1, 1, 1],
        "isMarried": [1, 0, 0],
        "isNoble": [1, 0, 0],
        "numDeadRelations": [1, 0, 0],
        "boolDeadRelations": [1, 0, 0],
        "isPopular": [1, 0, 0],
        "popularity": [0.605531171, 0.89632107, 0.043478261],
        "target": [1, 0, 0],
    }
    df = pd.DataFrame(data)
    X_train = df.drop(columns=["target"])
    y_train = df["target"].values
    return X_train, y_train


@pytest.fixture
def sample_pipeline():
    return create_pipeline()


def test_create_model_returns_fitted_objects(sample_data, sample_pipeline):
    X_train, y_train = sample_data
    model, search = create_model(X_train, y_train, sample_pipeline, cv=2)
    assert hasattr(model, "fit")  # Check if model is fitted
    assert hasattr(
        search, "cv_results_"
    )  # Check if RandomizedSearchCV has cv_results_ attribute


def test_create_model_type_error_with_invalid_data(sample_pipeline):
    with pytest.raises(TypeError):
        create_model("invalid data type", np.array([0, 1, 0]), sample_pipeline)

if __name__ == "__main__":
    pytest.main()
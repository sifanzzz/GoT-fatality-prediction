import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_pipeline import create_pipeline


# test set up
@pytest.fixture
def setup_data():
    # Assuming the structure of the DataFrame based on the image you provided
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
    }
    df = pd.DataFrame(data)
    return df


def test_pipeline_structure():
    """
    Test if the pipeline is created with the correct structure and transformers.
    """
    pipeline = create_pipeline()
    assert len(pipeline.transformers) == 3  # Checking if there are three transformers


def test_numeric_transformation(setup_data):
    """
    Test if numeric features are correctly scaled.
    """
    pipeline = create_pipeline()
    transformed = pipeline.fit_transform(setup_data)

    # Check if transformed data is not the same as original (since it's scaled)
    assert not np.array_equal(transformed[:, 0], setup_data["dateOfBirth"])


def test_categorical_transformation(setup_data):
    """
    Test if categorical features are correctly one-hot encoded.
    """
    pipeline = create_pipeline()
    transformed = pipeline.fit_transform(setup_data)

    # Check if the transformed data has more columns than the original
    # due to one-hot encoding
    assert transformed.shape[1] > setup_data.shape[1]


def test_passthrough_features(setup_data):
    """
    Test if passthrough features remain unchanged.
    """
    pipeline = create_pipeline()
    transformed = pipeline.fit_transform(setup_data)
    transformed = pd.DataFrame(transformed, columns=pipeline.get_feature_names_out())
    # Check if passthrough features remain the same
    assert np.array_equal(transformed["passthrough__male"], setup_data["male"])
    assert np.array_equal(transformed["passthrough__book1"], setup_data["book1"])
    assert np.array_equal(
        transformed["passthrough__popularity"], setup_data["popularity"]
    )


if __name__ == "__main__":
    pytest.main()
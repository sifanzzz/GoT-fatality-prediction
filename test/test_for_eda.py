import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.visualize_correlation import visualize_correlation

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


# Test if the function run without error
def test_visualize_correlation(setup_data):
    # Call the function
    heatmap = visualize_correlation(setup_data)

    # Assertion to check if the function ran without errors
    assert heatmap is not None

    # Additional assertion to check if plt.gcf() is not None
    assert plt.gcf() is not None

    # Check if the color scale is present
    color_scale = heatmap.collections[0].get_array()
    assert color_scale is not None


if __name__ == "__main__":
    pytest.main()
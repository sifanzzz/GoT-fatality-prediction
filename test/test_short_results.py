import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.short_results import short_results
import pandas as pd
import numpy as np
import pytest
import pickle


@pytest.fixture
def setup_data():
      # Assuming the structure of the DataFrame based on the image you provided
       with open("./test/model.pkl", "rb") as f:
              model = pickle.load(f)
       return model

def test_input_error(setup_data):
    with pytest.raises(TypeError):
        short_results([1,2,3])

def test_short_results_returns_dataframe(setup_data):
    banana = short_results(setup_data)
    assert isinstance(banana, pd.DataFrame)

def test_short_results_correct_size(setup_data):
    banana = short_results(setup_data)
    assert banana.shape == (5,6)

if __name__ == "__main__":
    pytest.main()
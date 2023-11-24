import pandas as pd
import pytest
from short_results import short_results



test_model = #some RandomizedSearchCV output

def test_short_results_returns_dataframe():
    banana = short_results(test_model)
    assert isinstance(df, pd.DataFrame)

pytest tests/
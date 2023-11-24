import pandas as pd
import pytest
test_data


test_model = random_search

def test_short_results_returns_dataframe():
    df = short_results(test_model)
    assert isinstance(df, pd.DataFrame)

pytest tests/
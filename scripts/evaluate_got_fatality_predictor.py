import click
import os
import sys
from sklearn import set_config
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.short_results import short_results


@click.command()
@click.option("-d", "--test-data", type=str, help="Path to test data")
@click.option(
    "-m",
    "--models-from",
    type=str,
    help="Path to directory where the fitted best model object and fitted random search object",
)
@click.option(
    "-r",
    "--results-to",
    type=str,
    help="Path to directory where the plot and scv will be save at",
)
def main(test_data, models_from, results_to):
    """Evaluates the got fatality model and random search on the test data
    and saves the evaluation results."""
    set_config(transform_output="pandas")
    # Load test data
    test_df = pd.read_csv(test_data)
    y_test = test_df["target"]
    X_test = test_df.drop("target", inplace=False, axis=1)

    # Load random search
    with open(os.path.join(models_from, "random_search.pkl"), "rb") as f:
        random_search = pickle.load(f)

    # Load the best model
    with open(os.path.join(models_from, "best_model.pkl"), "rb") as f:
        best_model = pickle.load(f)

    # Compute predicted y for test set
    y_pred = best_model.predict(X_test)

    # Get result from random search object
    random_search_result = short_results(random_search)
    random_search_result.to_csv(
        os.path.join(results_to, "tables", "random_search_scores.csv"), index=False
    )

    # Compute report for the best model
    class_report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).T
    class_report.to_csv(
        os.path.join(results_to, "tables", "class_report.csv"), index=False
    )

    # Compute PR curve
    pr_curve = PrecisionRecallDisplay.from_estimator(
        best_model, X_test, y_test, name="GoT best LR model"
    )
    pr_curve.figure_.savefig(os.path.join(results_to, "figures", "pr_curve.png"))

    # Compute ROC curve
    roc_curve = RocCurveDisplay.from_estimator(
        best_model, X_test, y_test, name="GoT best LR model"
    )
    roc_curve.figure_.savefig(os.path.join(results_to, "figures", "roc_curve.png"))


if __name__ == "__main__":
    main()

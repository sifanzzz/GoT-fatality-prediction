import click
import os
import pickle
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model_train import create_model
from src.data_pipeline import create_pipeline


@click.command()
@click.option("-d", "--training-data", type=str, help="Path to training data")
@click.option(
    "-m",
    "--model-to",
    type=str,
    help="Path to directory where the model and random search object will be written to",
)
@click.option(
    "--cv",
    type=int,
    help="Number of time to run cv in random search, default 5",
    default=5,
)
def main(training_data, model_to, cv):
    """Fits a got fatality classifier to the training data with random search
    and saves the random search and best model object.
    """

    # Load training data
    train_df = pd.read_csv(training_data)

    # Split X and y from the data
    y_train = train_df["target"]
    X_train = train_df.drop("target", inplace=False, axis=1)

    # Create pipeline
    preprocessor = create_pipeline()

    # Create model and fit the best model by useing CV and random search
    best_model, random_search = create_model(
        X_train, y_train, preprocessor, save_path=None, cv=cv
    )

    # Save the best model
    with open(os.path.join(model_to, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    # Save random search
    with open(os.path.join(model_to, "random_search.pkl"), "wb") as f:
        pickle.dump(random_search, f)


if __name__ == "__main__":
    main()

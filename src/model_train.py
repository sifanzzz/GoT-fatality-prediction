from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import pickle
import os
import pandas as pd
import numpy as np


def create_model(X, y, pipeline, save_path=None, cv=5):
    """
    Creates and fits a Logistic Regression model using a specified pipeline for preprocessing
    and RandomizedSearchCV for hyperparameter tuning. Optionally saves the trained model to disk.

    Parameters:
    X (pd.DataFrame or np.ndarray): Features of the training data, expected to not to include a 'target' column.
    y (list, np.ndarray, pd.DataFrame): Array of holding the target of the training data.
    pipeline (ColumnTransformer): A scikit-learn ColumnTransformer for data preprocessing.
    save_path (str, optional): Path to save the trained model as a pickle file. If None, the model is not saved.
    cv (int): A int to indicate to run time of cross validation.

    Returns:
    tuple: A tuple containing the fitted pipeline (with Logistic Regression) and the RandomizedSearchCV object.

    Example:
    model, search = create_model(X_train, y_train, pipeline, save_path="./")

    print(model)
    print(search.cv_results_)
    """
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            "training data need to be in numpy array or pandas DataFrame, got %s",
            type(X).__name__,
        )

    if not isinstance(pipeline, ColumnTransformer):
        raise TypeError(
            "pipeline need to be in ColumnTransformer, got %s" % type(pipeline)
        )

    # run random_search for the best model para with CV
    param_dist = {
        "logisticregression__class_weight": [None, "balanced"],
        "logisticregression__C": np.logspace(-7, 5, 20),
        "logisticregression__max_iter": [100, 500, 1000, 1500, 2000],
    }
    logreg_pipe = make_pipeline(pipeline, LogisticRegression(random_state=123))
    random_search = RandomizedSearchCV(
        logreg_pipe,
        param_dist,
        n_jobs=-1,
        n_iter=40,
        cv=cv,
        scoring="f1",
        return_train_score=True,
        random_state=123,
    )
    random_search.fit(X, y)

    # load the best model with best para with all the training data
    best_params = random_search.best_params_
    logreg_pipe.set_params(**best_params)
    logreg_pipe.fit(X, y)

    if save_path:
        with open(os.path.join(save_path, "model.pkl"), "wb") as f:
            pickle.dump(logreg_pipe, f)

    return logreg_pipe, random_search

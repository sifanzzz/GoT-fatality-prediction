import pandas as pd

def short_results(cv_results):
    """
    Summarize the cross validation attribute of a RandomizedSearchCV model containing a logistic regressor.

    Creates a dataframe with six columns and four rows, reporting the model parameters C, class weights, and maximum iterations, 
    along with the mean fit time, train score, and test score for each of the four random parameter combination trials having
    highest mean test score.

    Parameters:
    ----------
    
    model : <class 'sklearn.model_selection._search.RandomizedSearchCV'>
        The model resulting from a RandomizedSearchCV call. 

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with six columns:
        - 'param_logisticregression__C': Lists the LR parameter C used in each trial.
        - 'param_logisticregression__max_iter': Lists the LR parameter max_iter used in each trial.
        - 'param_logisticregression__class_weight' : Lists the LR parameter class_weight used in each trial.
        - 'mean_test_score' : Lists the mean test score for each trial.
        - 'mean_fit_time' : Lists the mean fit time for each trial.
        - 'mean_train_score' : Lists the mean train score for each trial.
        
    Examples:
    --------
    >>> import pandas as pd
    >>> model = random_search  # Replace 'random_search' with your RandomizedSearchCV output
    >>> result = short_results(model)
    >>> print(result)
    
    Notes:
    -----
    This function tailors the output of model.cv_results_, which contains a wider variety of statistics for all 
    of the random trials undertaken by RandomSearchCV. You are advised to make yourself aware of the information
    contained in this full output before using short_results().

    """

    assert(type(cv_results) == dict)
    results = pd.DataFrame(cv_results)
    sorted_results = results.sort_values(by="mean_test_score", ascending=False).reset_index(drop=True)
    return sorted_results.loc[:4,["param_logisticregression__C",
                        "param_logisticregression__max_iter",
                        "param_logisticregression__class_weight",
                        "mean_test_score",
                        "mean_fit_time",
                        "mean_train_score"]]
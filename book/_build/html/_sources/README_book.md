# Team 6 Milestone1 Project:

# Game of Throne Characters Fatality Predictor

-   Author: Thomas Jian, Ian MacCarthy, Arturo Rey, Sifan Zhang

Milestone1 project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## Overview

This project aims to predict the mortality of characters in "Game of Thrones" using machine learning models. The model is designed to analyze data related to characters in the book series "Game of Thrones" and predict whether a character is likely to survive or die in the story line.

Fatality prediction in this context is a binary classification task, where the outcome is categorized as either `isDead=0` (indicating survive) or `isDead=1` (indicating dead). By training on the data, we aim to create a model capable of providing insights into the fate of characters based on various features.

## Model Comparison

Our initial model comparison involved three classifiers: DummyClassifier, Logistic Regression (LR), and Support Vector Classifier (SVC). From the evaluation metrics, LR emerged as the top-performing model. Therefore, we proceeded hyperparameter optimization for LR to enhance its predictive capabilities.

## Test Set Evaluation

The optimized LR model was then evaluated on a test set of 390 instances, resulting in the following key metrics:

-   **Accuracy:** 0.63
-   **Precision, Recall, and F1-score for Each Class:**
    -   Dead (Class 1): Precision 0.85, Recall 0.62, F1-score 0.72 (Support: 294)
    -   Survive (Class 0): Precision 0.38, Recall 0.68, F1-score 0.47 (Support: 96)
-   **Macro Average:** Precision 0.61, Recall 0.65, F1-score 0.59
-   **Weighted Average:** Precision 0.73, Recall 0.63, F1-score 0.66

These metrics provide a comprehensive understanding of the model's performance in predicting character survival or fatality.

## Next Steps

While LR exhibited promising results, further refinement and exploration of additional features may be necessary to enhance predictive accuracy. Future considerations might involve more sophisticated models to improve performance.

## Usage

To run the project via docker, run the following from the root of this repository:

```bash
docker compose up
```
Open your browser and type the following into the address bar:

```bash
localhost:8890
```

At the end of your session, hit **cntrl+C** at the command line and then run the following:

```bash
docker compose rm 
```

If you don't want to use docker, then the first time running the project, run the following from the root of this repository:

``` bash
conda env create --file 522env.yaml
```

To run the analysis, run the following from the root of this repository:

``` bash
conda activate GoT-fatality-prediction
jupyter lab 
```


Open `got_fatality_predictor.ipynb` in Jupyter Lab and under the "Kernel" menu click "Restart Kernel and Run All Cells".



To run the tests, run the following from the root of this repository:

``` bash
pytest test/
```


To visualize the notebook in a browser, go to the following link:

<https://ianm99.github.io/Team-6-publishing/index.html>


## Reference

1.  Data Society. 2016. Requests: Game of Thrones. <https://data.world/data-society/game-of-thrones>
2.  Joel Östblom. 2023. DSCI531 Course Notes. <https://pages.github.ubc.ca/MDS-2023-24/DSCI_531_viz-1_students/lectures/4-eda.html>
3.  Varada Kolhatkar. 2023. DSCI571 Course Notes . <https://pages.github.ubc.ca/MDS-2023-24/DSCI_571_sup-learn-1_students/lectures/00_motivation-course-information.html>
4.  Joel Östblom. 2023. DSCI573 Course Notes. <https://pages.github.ubc.ca/MDS-2023-24/DSCI_573_feat-model-select_students/README.html>

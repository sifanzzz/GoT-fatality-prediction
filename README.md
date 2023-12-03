# Team 6 Milestone Project:

# Game of Throne Characters Fatality Predictor

-   Author: Thomas Jian, Ian MacCarthy, Arturo Rey, Sifan Zhang

Milestone project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

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

## Test Summary

These metrics provide a comprehensive understanding of the model's performance in predicting character survival or fatality. Overall the model's accuracy is fairly unimpressive, correctly predicting the fate of a character in only about half of all cases. While this might seem a bit disappointing, it did not particularly surprise or discourage us: George R. R. Martin is a celebrated author and master story teller, and the fact that we can't easily predict a whether a character will survive based on their attributes is a testament to the quality of his writing rather than the inadequacy of our model.

## Next Steps

Further refinement and exploration of additional features may be necessary to enhance predictive accuracy. Future considerations might involve more sophisticated models to improve performance.

## Report

The final report can be found [here](https://ianm99.github.io/Milestone-3/got_fatality_predictor_book.html).

To visualize the notebook in a browser, go to the following link:

<https://ianm99.github.io/Team-6-publishing/index.html>

## Dependencies

[Docker](https://www.docker.com/) is a container solution used to manage the software dependencies for this project. The Docker image used for this project is based on the quay.io/jupyter/minimal-notebook:2023-11-19. Additioanal dependencies are specified int the [Dockerfile](https://github.com/UBC-MDS/GoT-fatality-prediction/blob/main/Dockerfile).

## Usage

#### Setup

1.  **Setting up for to run this analysis via docker**

-   [Install](https://www.docker.com/get-started/) and launch Docker on your computer.
-   Clone this GitHub repository.

after setting up, run the following from the root of this repository:

``` bash
docker compose up
```

Open your browser and type the following into the address bar:

``` bash
localhost:8890
```

Open the directory

click on `work/`

2.  **Setting up to running this analysis via conda environment.**

If you don't want to use docker, then the first time running the project, run the following from the root of this repository:

``` bash
conda env create --file 522env.yaml -n GoT-fatality-prediction
```

Use the \`GoT-fatality-prediction\` environment and open the project with jupyter lab

``` bash
conda activate GoT-fatality-prediction
jupyter lab 
```

#### Run the project

After setting up with one of the above method, to run the analysis, run the following from the root of this repository:

``` bash
# download and extract data

python scripts/download_data.py \
    --url=https://raw.githubusercontent.com/TheMLGuy/Game-of-Thrones-Dataset/master/character-predictions.csv \
    --write_to=data/processed/preprocessed.csv

# calculate the NaN percentage of the data

python src/calculate_missing_percentage.py \
    --file_path=data/character-predictions_pose.csv \
    --output_file=data/tables/calculate_missing_percentage.csv \
    
# preprocess and split the data into train and test

python scripts/preprocess.py \
    --input_filepath=data/processed/preprocessed.csv \
    --output_filepath_train=data/processed/train.csv \
    --output_filepath_test=data/processed/test.csv \
    --seed=123

# generate correlation heatmap of features

python scripts/visualize_correlation.py \
    --file_path=data/processed/train.csv \
    --output_file=results/figures/correlation_heatmap.png \
    
# train model, save model object and random search object

python scripts/fit_got_fatality_classifier.py \
    --training-data data/processed/train.csv \
    --model-to results/models/

# evaluate model on test data and save results

python scripts/evaluate_got_fatality_predictor.py \
    --test-data data/processed/test.csv \
    --models-from results/models/ \
    --results-to results/
```

## Clean up

To shut down the container and clean up the resources, type Cntrl + C in the terminal where you launched the container, and then type docker docker compose down

``` bash
docker compose down
```

## Reference

1.  Data Society. 2016. Requests: Game of Thrones. <https://data.world/data-society/game-of-thrones>
2.  Joel Östblom. 2023. DSCI531 Course Notes. <https://pages.github.ubc.ca/MDS-2023-24/DSCI_531_viz-1_students/lectures/4-eda.html>
3.  Varada Kolhatkar. 2023. DSCI571 Course Notes . <https://pages.github.ubc.ca/MDS-2023-24/DSCI_571_sup-learn-1_students/lectures/00_motivation-course-information.html>
4.  Joel Östblom. 2023. DSCI573 Course Notes. <https://pages.github.ubc.ca/MDS-2023-24/DSCI_573_feat-model-select_students/README.html>

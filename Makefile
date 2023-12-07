all: docs/index.html

# download and extract data
data/processed/preprocessed.csv: scripts/download_data.py
	python scripts/download_data.py \
		--url=https://raw.githubusercontent.com/TheMLGuy/Game-of-Thrones-Dataset/master/character-predictions.csv \
		--write_to=data/processed/preprocessed.csv

# calculate the NaN percentage of the data
results/tables/calculate_missing_percentage.csv: scripts/calculate_missing_percentage.py data/character-predictions_pose.csv
	python scripts/calculate_missing_percentage.py \
		--file_path=data/character-predictions_pose.csv \
		--output_file=results/tables/calculate_missing_percentage.csv


# preprocess and split the data into train and test
data/processed/train.csv data/processed/test.csv: scripts/preprocess.py data/processed/preprocessed.csv
	python scripts/preprocess.py \
		--input_filepath=data/processed/preprocessed.csv \
		--output_filepath_train=data/processed/train.csv \
		--output_filepath_test=data/processed/test.csv \
		--seed=123

# generate correlation heatmap of features
results/figures/correlation_heatmap.png: scripts/visualize_correlation.py data/processed/train.csv data/processed/test.csv
	python scripts/visualize_correlation.py \
		--file_path=data/processed/train.csv \
		--output_file=results/figures/correlation_heatmap.png 

# train model, save model object and random search object
results/models/best_model.pkl results/models/random_search.pkl: scripts/fit_got_fatality_classifier.py data/processed/train.csv 
	python scripts/fit_got_fatality_classifier.py \
		--training-data data/processed/train.csv \
		--model-to results/models/

# evaluate model on test data and save results
results/figures/pr_curve.png results/figures/roc_curve.png results/tables/class_report.csv results/tables/random_search_scores.csv: scripts/evaluate_got_fatality_predictor.py data/processed/test.csv results/models/best_model.pkl results/models/random_search.pkl
	python scripts/evaluate_got_fatality_predictor.py \
		--test-data data/processed/test.csv \
		--models-from results/models/ \
		--results-to results/

# build notebook form all the results 
book/build/html/index.html: book/got_fatality_predictor_book.ipynb \
book/_toc.yml \
book/_config.yml \
results/ \
results/figures/pr_curve.png \
results/figures/correlation_heatmap.png \
results/figures/roc_curve.png \
results/tables/calculate_missing_percentage.csv \
results/tables/class_report.csv \
results/tables/random_search_scores.csv
	jupyter-book build book/

# copy all the html file to docs so github page can work
docs/index.html: book/build/html/index.html
	mkdir -p docs/ && cp -r book/_build/html/* docs/ && touch docs/.nojekyll

clean:
	rm -f results/figures/pr_curve.png \
		results/figures/correlation_heatmap.png \
		results/figures/roc_curve.png \
		results/tables/calculate_missing_percentage.csv \
		results/tables/class_report.csv \
		results/tables/random_search_scores.csv \
		results/models/best_model.pkl \
		results/models/random_search.pkl
	rm -rf report/_build 
	rm -rf docs/
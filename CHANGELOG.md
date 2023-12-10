# Peer Review Edits

Problem: There was an extraneous link in the README.md leading to a broken repo where the milestone 2 website used to live. It has now been removed to eliminate confusion since we already have the report as a jupyter book in a new repo

<https://github.com/UBC-MDS/Team6/blob/3a655eacf42a1a54a1074f9fc662e245055baf6b/README.md#L44>

Probelm: There was some confusion over what the features actually represent, since their names are not immediately informative to someone unfamiliar with the project. A section was added to the report providing a brief interpretation of each feature.

<https://vscode.dev/github/UBC-MDS/Team6/blob/main/book/got_fatality_predictor_book.ipynb#C9>


Problem: There were some variables that were left in the EDA that did not make sense to the model, and were not mentioned in the feature explanations. These variables were `alive`, `pred`, and `actual` They were removed as requested by the reviewers.

<https://github.com/UBC-MDS/GoT-fatality-prediction/blob/changelog-update/scripts/preprocess.py>

Problem: There was a script that did not follow the proper name convention. The reviewers asked to either remove it or rename it if it was important. It was renamed from `ians_script.py` to `results_show.py` in order for it to be more telling of what it does.

<https://github.com/UBC-MDS/GoT-fatality-prediction/blob/changelog-update/src/results_show.py>
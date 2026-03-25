# Random_Forest_Self-Training-_Do_Confidence_Thresholds_Affect_Accuracy
Confidence-threshold–based self-training with a Random Forest on a 2D binary classification task, using a synthetic make_moons dataset of 1,000 samples where 100 are labeled and 700 are treated as unlabeled examples.

requirements:
Python 3.x
NumPy
Matplotlib
scikit-learn (for datasets, model_selection, preprocessing, ensemble, metrics)
A Jupyter environment (e.g., Jupyter Notebook or Google Colab) to run the .ipynb file

Problem setup:
Dataset:
make_moons with noise, split into train/test with stratification.
Labeled vs unlabeled: 100 labeled training points, 700 unlabeled.
Base model: RandomForestClassifier with 200 trees, default depth, random_state=42.
Baseline: Train only on the 100 labeled points and evaluate on the test set.

How to run:
Install requirements (e.g. scikit-learn, matplotlib, numpy).
Open the notebook in Jupyter/Colab.
Run all cells in order to:
Generate the data.
Train the baseline Random Forest.
Run self-training at multiple thresholds.
Visualize before/after decision boundaries.

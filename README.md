# Random_Forest_Self-Training-_Do_Confidence_Thresholds_Affect_Accuracy

Confidence-threshold–based self-training with a Random Forest on a 2D binary classification task, using a synthetic `make_moons` dataset of 1,000 samples where 100 are labeled and 700 are treated as unlabeled examples. [page:8][file:7]

## Requirements

- Python 3.x  
- NumPy  
- Matplotlib  
- scikit-learn (datasets, model_selection, preprocessing, ensemble, metrics)  
- A Jupyter environment (Jupyter Notebook or Google Colab) to run the `.ipynb` file [page:8]

## Problem setup

- Dataset: `make_moons` with noise, split into train/test with stratification.  
- Labeled vs unlabeled: 100 labeled training points, 700 unlabeled.  
- Base model: `RandomForestClassifier` with 200 trees, default depth, `random_state=42`.  
- Baseline: Train only on the 100 labeled points and evaluate on the test set. [page:8][file:7]

## How to run

1. Install requirements (e.g. `scikit-learn`, `matplotlib`, `numpy`).  
2. Open the notebook in Jupyter or Google Colab.  
3. Run all cells in order to:  
   - Generate the data.  
   - Train the baseline Random Forest.  
   - Run self-training at multiple confidence thresholds (0.8, 0.9, 0.95).  
   - Visualize before/after decision boundaries. [page:8][file:7]
  
## Learning objectives

- Understand Random Forest classification on non-linear 2D data.  
- Implement and experiment with self-training using confidence thresholds.  
- Interpret how pseudo-labels from unlabeled data influence accuracy. [file:7]

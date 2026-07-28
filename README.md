# iris-xgboost-classifier
# Iris Species Classification with XGBoost

A simple end-to-end multi-class classification project using XGBoost on the
classic Iris dataset. Built as part of hands-on practice with tree-based
models, feature analysis, and model evaluation.

## What it does

- Loads the Iris dataset (150 samples, 3 species, 4 numeric features)
- Checks feature correlation to understand relationships between measurements
- Trains an `XGBClassifier` on a 75/25 train-test split
- Evaluates performance with accuracy, precision, recall, and F1-score
- Saves the trained model to disk for reuse

## Results

- **Accuracy:** 1.00 on the test set
- **Precision / Recall / F1:** 1.00 across all three classes

Iris is a small, well-separated dataset, so perfect accuracy here reflects
the dataset's simplicity rather than exceptional tuning. Confirmed via the
full classification report (no class had hidden errors).

## Key takeaways

- **Petal length and petal width are highly correlated (~0.96)** — makes
  intuitive sense, since larger petals tend to be both longer and wider.
- **Sepal width behaves differently** from the other features (weak/negative
  correlation with everything else) — the classic quirk that makes it the
  least useful single feature for species separation.
- **XGBoost doesn't require dropping correlated features.** Since it's a
  tree-based ensemble, each split just picks whichever feature helps most
  at that node — multicollinearity mainly matters for linear/logistic
  models, not tree-based ones.
- **Learning rate matters.** An overly high learning rate (e.g. 100) causes
  the model to overshoot; a small rate (e.g. 0.01–0.1) with enough
  estimators trains a stable model.

## How to run

```bash
pip install -r requirements.txt
python iris_classifier.py
```

## Files

- `iris_classifier.py` — main training/evaluation script
- `requirements.txt` — dependencies
- `iris_model.json` — saved trained model (generated after running the script)

## Tech stack

Python, pandas, scikit-learn, XGBoost

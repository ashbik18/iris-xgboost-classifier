"""
Iris Species Classification with XGBoost

A simple multi-class classification pipeline using XGBoost on the classic
Iris dataset. Includes feature correlation analysis and full evaluation
metrics (accuracy, precision, recall, f1).
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data():
    """Load Iris dataset as a pandas DataFrame."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


def explore_features(X):
    """Print feature correlation matrix to check for multicollinearity."""
    print("Feature Correlation Matrix:")
    print(X.corr())
    print()
    # Note: XGBoost is tree-based and handles correlated features fine,
    # so no need to drop any here. This check matters more for linear
    # models (e.g. logistic regression) where multicollinearity can
    # destabilize coefficients.


def train_model(X_train, y_train):
    """Train an XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,   # low learning rate needs more trees to converge;
                             # 0.1 is a reasonable default for this dataset size
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def main():
    X, y = load_data()

    explore_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save trained model for reuse without retraining
    model.save_model('iris_model.json')
    print("\nModel saved to iris_model.json")


if __name__ == "__main__":
    main()

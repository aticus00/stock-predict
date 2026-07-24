"""Walk-forward evaluation and next-day up-probability prediction."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


def _make_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)


def evaluate_walk_forward(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    n_splits = min(n_splits, len(X) - 1)
    if n_splits < 2:
        return float("nan")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model = _make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], preds))
    return sum(scores) / len(scores)


def train_final_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = _make_model()
    model.fit(X, y)
    return model


def predict_latest(model: RandomForestClassifier, latest_row: pd.DataFrame) -> float:
    proba = model.predict_proba(latest_row)
    up_index = list(model.classes_).index(1)
    return float(proba[0, up_index])

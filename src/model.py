"""Walk-forward evaluation, next-day up-probability, and N-day price forecast."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from features import build_features


def _make_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)


def _make_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0)


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


def train_return_regressor(X: pd.DataFrame, y_return: pd.Series) -> RandomForestRegressor:
    model = _make_regressor()
    model.fit(X, y_return)
    return model


def forecast_future_prices(
    df: pd.DataFrame, regressor: RandomForestRegressor, n_days: int
) -> pd.Series:
    """Iteratively apply the regressor's predicted next-day return to project
    a rough N-day price path. Volume is held at its last observed value since
    future volume is unknown. Error compounds with each step — illustrative
    only, not a reliable forecast."""
    sim_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    last_volume = sim_df["Volume"].iloc[-1]

    forecast_dates, forecast_prices = [], []
    for _ in range(n_days):
        _, _, _, latest_row = build_features(sim_df)
        predicted_return = float(regressor.predict(latest_row)[0])
        next_close = sim_df["Close"].iloc[-1] * (1 + predicted_return)
        next_date = sim_df.index[-1] + pd.tseries.offsets.BDay(1)

        sim_df.loc[next_date] = {
            "Open": next_close,
            "High": next_close,
            "Low": next_close,
            "Close": next_close,
            "Volume": last_volume,
        }
        forecast_dates.append(next_date)
        forecast_prices.append(next_close)

    return pd.Series(forecast_prices, index=pd.Index(forecast_dates))

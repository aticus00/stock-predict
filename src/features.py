"""Feature engineering from daily OHLCV bars."""
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ma5_ma20_gap",
    "volatility_20",
    "rsi_14",
    "macd_hist",
    "volume_ratio_20",
    "momentum_5",
    "momentum_10",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return (X, y_up, y_return, latest_row).

    y_up: next-day up/down classification label (0/1).
    y_return: next-day pct return regression label, used for multi-day forecasting.
    latest_row: unlabeled feature row for the most recent bar (used for live prediction).
    """
    close, volume = df["Close"], df["Volume"]

    feat = pd.DataFrame(index=df.index)
    feat["return_1d"] = close.pct_change()
    ma5, ma20, ma60 = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    feat["ma5_ratio"] = close / ma5
    feat["ma20_ratio"] = close / ma20
    feat["ma60_ratio"] = close / ma60
    feat["ma5_ma20_gap"] = ma5 / ma20 - 1
    feat["volatility_20"] = feat["return_1d"].rolling(20).std()
    feat["rsi_14"] = _rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - macd_signal) / close
    feat["volume_ratio_20"] = volume / volume.rolling(20).mean()
    feat["momentum_5"] = close / close.shift(5) - 1
    feat["momentum_10"] = close / close.shift(10) - 1

    target_up = (close.shift(-1) > close).astype(int)
    target_return = close.shift(-1) / close - 1

    latest_row = feat.iloc[[-1]]
    labeled = feat.iloc[:-1]

    valid = labeled.notna().all(axis=1)
    X = labeled.loc[valid]
    y_up = target_up.iloc[:-1].loc[valid]
    y_return = target_return.iloc[:-1].loc[valid]

    return X, y_up, y_return, latest_row

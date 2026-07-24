"""Library compatibility smoke test for the Python 3.14 / pandas 3 / numpy 2.5 stack."""
import sys

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import plotly
import streamlit
from sklearn.ensemble import RandomForestClassifier

print("python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("streamlit:", streamlit.__version__)
print("plotly:", plotly.__version__)
print("finance-datareader:", fdr.__version__)

df = fdr.DataReader("005930", "2024-01-01")
assert not df.empty, "FinanceDataReader returned an empty DataFrame"
assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
print("fdr.DataReader OK:", df.shape, "rows, columns:", list(df.columns))

df["ma5"] = df["Close"].rolling(5).mean()
assert df["ma5"].notna().sum() > 0
print("pandas rolling OK")

rng = np.random.default_rng(0)
X = rng.random((50, 4))
y = rng.integers(0, 2, size=50)
clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(X, y)
proba = clf.predict_proba(X[:1])
assert proba.shape == (1, 2)
print("sklearn RandomForestClassifier OK:", proba)

print("\nSMOKE TEST PASSED")

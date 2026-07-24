"""KRX market data access via FinanceDataReader, cached for Streamlit reruns."""
import datetime as dt
import os

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import FinanceDataReader as fdr
import pandas as pd
import streamlit as st

DISPLAY_WINDOWS = {"6개월": 0.5, "1년": 1, "3년": 3}

# Model training always uses this much history regardless of the chart's
# display window, so shortening the chart doesn't starve the model of data.
TRAIN_YEARS = 3


@st.cache_data(ttl=3600)
def fetch_ohlcv(code: str, years: float = TRAIN_YEARS) -> pd.DataFrame:
    start = dt.date.today() - dt.timedelta(days=int(365 * years))
    df = fdr.DataReader(code.zfill(6), start.isoformat())
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


@st.cache_data(ttl=86400)
def get_krx_listing() -> pd.DataFrame:
    listing = fdr.StockListing("KRX")
    return listing[["Code", "Name"]].dropna()

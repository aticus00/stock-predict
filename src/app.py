"""Streamlit dashboard: KRX stock chart + next-day up-probability + N-day price forecast."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import DISPLAY_WINDOWS, TRAIN_YEARS, fetch_ohlcv, get_krx_listing
from features import build_features
from model import (
    evaluate_baseline_walk_forward,
    evaluate_walk_forward,
    forecast_future_prices,
    predict_latest,
    train_final_model,
    train_return_regressor,
)

DISCLAIMER = "⚠️ 이 도구는 간단한 통계 모델(RandomForest)이며 투자 조언이 아닙니다. 참고용으로만 사용하세요."
FORECAST_DAYS = {"5일": 5, "10일": 10, "20일": 20}


def build_chart(display_label: str, df: pd.DataFrame, display_years: float, forecast=None) -> go.Figure:
    ma20 = df["Close"].rolling(20).mean()
    ma60 = df["Close"].rolling(60).mean()

    cutoff = df.index[-1] - pd.Timedelta(days=int(365 * display_years))
    visible = df.index >= cutoff

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[visible], y=df["Close"][visible], name="종가"))
    fig.add_trace(go.Scatter(x=df.index[visible], y=ma20[visible], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index[visible], y=ma60[visible], name="MA60"))
    if forecast is not None and len(forecast) > 0:
        connect_x = [df.index[-1], *forecast.index]
        connect_y = [df["Close"].iloc[-1], *forecast.values]
        fig.add_trace(
            go.Scatter(
                x=connect_x,
                y=connect_y,
                name=f"예측 ({len(forecast)}일)",
                line=dict(dash="dot", color="orange"),
            )
        )
    fig.update_layout(
        title=dict(text=f"{display_label} 일별 시세", font=dict(size=20)),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=16),
        legend=dict(font=dict(size=15)),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
    )
    return fig


st.set_page_config(page_title="한국 주식 예측 대시보드", layout="wide")
st.markdown(
    """
    <style>
    html { font-size: 130% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📈 한국 주식 상승 가능성 대시보드")
st.warning(DISCLAIMER)

st.subheader("종목 선택")
listing = None
try:
    listing = get_krx_listing()
except Exception:
    st.warning("종목 목록을 불러오지 못했습니다. 종목 코드를 직접 입력하세요.")

col_search, col_select = st.columns(2)
if listing is not None:
    query = col_search.text_input("종목명 검색", "삼성전자")
    matches = listing[listing["Name"].str.contains(query, case=False, na=False)] if query else listing.iloc[0:0]
    if not matches.empty:
        options = [f"{row.Name} ({row.Code})" for row in matches.itertuples()]
        choice = col_select.selectbox("종목 선택", options)
        code = choice.split("(")[-1].rstrip(")")
        display_label = choice
    else:
        col_select.info("검색 결과가 없습니다. 종목 코드를 직접 입력하세요.")
        code = col_select.text_input("종목 코드 (6자리)", "005930")
        match = listing[listing["Code"] == code.zfill(6)]
        display_label = f"{match.iloc[0].Name} ({code})" if not match.empty else code
else:
    code = col_search.text_input("종목 코드 (6자리)", "005930")
    display_label = code

col_period, col_forecast = st.columns(2)
display_label_period = col_period.selectbox("차트 표시 기간", list(DISPLAY_WINDOWS.keys()), index=0)
display_years = DISPLAY_WINDOWS[display_label_period]
forecast_label = col_forecast.selectbox("예측 기간", list(FORECAST_DAYS.keys()), index=1)
n_forecast_days = FORECAST_DAYS[forecast_label]
st.caption(f"모델 학습에는 표시 기간과 무관하게 항상 최근 {TRAIN_YEARS}년치 데이터를 사용합니다.")

if not code:
    st.stop()

try:
    df = fetch_ohlcv(code)
except Exception as e:
    st.error(f"데이터를 가져오지 못했습니다: {e}")
    st.stop()

if df.empty:
    st.error("해당 종목의 데이터가 없습니다. 종목 코드를 확인해주세요.")
    st.stop()

run_prediction = st.button("예측 실행", type="primary")

accuracy = baseline_accuracy = up_probability = forecast = None
if run_prediction:
    X, y_up, y_return, latest_row = build_features(df)
    if len(X) < 30:
        st.error("예측을 위한 데이터가 충분하지 않습니다.")
    else:
        with st.spinner("모델 학습 중..."):
            accuracy = evaluate_walk_forward(X, y_up)
            baseline_accuracy = evaluate_baseline_walk_forward(y_up)
            classifier = train_final_model(X, y_up)
            up_probability = predict_latest(classifier, latest_row)

            regressor = train_return_regressor(X, y_return)
            forecast = forecast_future_prices(df, regressor, n_forecast_days)

st.plotly_chart(build_chart(display_label, df, display_years, forecast), width="stretch")

if accuracy is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("모델 정확도 (walk-forward)", f"{accuracy:.1%}")
    col2.metric("단순 다수결 베이스라인", f"{baseline_accuracy:.1%}")
    col3.metric("내일 상승 가능성", f"{up_probability:.1%}")
    if accuracy <= baseline_accuracy + 0.02:
        st.caption(
            "모델 정확도가 '그냥 다수결로 찍기'와 비슷하거나 낮습니다. 가격·거래량 데이터만으로는 "
            "다음날 방향을 안정적으로 맞히기 어렵다는 뜻이며, 이는 모델의 결함이 아니라 이 문제 자체의 "
            "근본적인 한계입니다."
        )
    st.caption(f"주황 점선은 {n_forecast_days}일치 예측 종가입니다. 하루 예측을 반복 적용한 값이라 기간이 길수록 오차가 누적됩니다.")
    st.info(DISCLAIMER)

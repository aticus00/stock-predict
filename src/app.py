"""Streamlit dashboard: KRX stock chart + next-day up-probability estimate."""
import plotly.graph_objects as go
import streamlit as st

from data import LOOKBACK_YEARS, fetch_ohlcv, get_krx_listing
from features import build_features
from model import evaluate_walk_forward, predict_latest, train_final_model

DISCLAIMER = "⚠️ 이 도구는 간단한 통계 모델(RandomForest)이며 투자 조언이 아닙니다. 참고용으로만 사용하세요."

st.set_page_config(page_title="한국 주식 예측 대시보드", layout="wide")
st.title("📈 한국 주식 상승 가능성 대시보드")
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("종목 선택")
    try:
        listing = get_krx_listing()
        query = st.text_input("종목명 검색", "삼성전자")
        matches = listing[listing["Name"].str.contains(query, case=False, na=False)] if query else listing.iloc[0:0]
        if not matches.empty:
            options = [f"{row.Name} ({row.Code})" for row in matches.itertuples()]
            choice = st.selectbox("종목 선택", options)
            code = choice.split("(")[-1].rstrip(")")
        else:
            st.info("검색 결과가 없습니다. 종목 코드를 직접 입력하세요.")
            code = st.text_input("종목 코드 (6자리)", "005930")
    except Exception:
        st.warning("종목 목록을 불러오지 못했습니다. 종목 코드를 직접 입력하세요.")
        code = st.text_input("종목 코드 (6자리)", "005930")

    period_label = st.selectbox("조회 기간", list(LOOKBACK_YEARS.keys()), index=1)
    years = LOOKBACK_YEARS[period_label]

if not code:
    st.stop()

try:
    df = fetch_ohlcv(code, years)
except Exception as e:
    st.error(f"데이터를 가져오지 못했습니다: {e}")
    st.stop()

if df.empty:
    st.error("해당 종목의 데이터가 없습니다. 종목 코드를 확인해주세요.")
    st.stop()

st.subheader(f"{code} 일별 시세")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="종가"))
fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="MA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(60).mean(), name="MA60"))
fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

if st.button("예측 실행", type="primary"):
    X, y, latest_row = build_features(df)
    if len(X) < 30:
        st.error("예측을 위한 데이터가 충분하지 않습니다. 조회 기간을 늘려주세요.")
    else:
        with st.spinner("모델 학습 중..."):
            accuracy = evaluate_walk_forward(X, y)
            model = train_final_model(X, y)
            up_probability = predict_latest(model, latest_row)

        col1, col2 = st.columns(2)
        col1.metric("백테스트 정확도 (walk-forward)", f"{accuracy:.1%}")
        col2.metric("내일 상승 가능성", f"{up_probability:.1%}")
        st.info(DISCLAIMER)

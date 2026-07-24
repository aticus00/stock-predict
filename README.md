# stock-predict

한국(KOSPI/KOSDAQ) 종목의 과거 데이터를 바탕으로 다음날 상승 가능성(확률)과 N일치 미래 가격 흐름을 추정해 보여주는 Streamlit 대시보드입니다.

> ⚠️ **이 도구는 간단한 통계 모델(RandomForest)이며 투자 조언이 아닙니다.** 예측 확률과 백테스트 정확도를 참고용으로만 사용하세요.

## 설치

Python 3.14가 필요합니다.

```bash
py -3.14 -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
```

## 실행

```bash
./.venv/Scripts/streamlit run src/app.py
```

## 동작 방식

1. 사이드바에서 종목을 검색/선택하면 [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader)로 일별 시세를 가져옵니다.
2. 종가와 이동평균선 차트를 바로 보여줍니다.
3. "예측 실행"을 누르면 이동평균/RSI/변동성 등 특징을 계산하고, `TimeSeriesSplit` 기반 walk-forward 검증으로 백테스트 정확도를 측정한 뒤, 전체 데이터로 학습한 분류 모델이 다음날 상승 확률을 예측합니다.
4. 동시에 회귀 모델이 다음날 수익률을 예측하고, 이를 선택한 기간(5/10/20일)만큼 반복 적용해 미래 종가 경로를 점선으로 이어 그립니다. 하루 예측을 누적하는 방식이라 기간이 길어질수록 오차가 커지는 참고용 추정치입니다.

## 향후 아이디어

- 여러 종목 비교, 예측 이력 저장, 다른 모델(LSTM 등) 실험 — 현재는 다루지 않는 범위입니다.

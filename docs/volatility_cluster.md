# 뉴스 기반 변동률 클러스터 모델

## 개요

`shared/cluster/` 패키지에 구현된 모듈이다.

"지금 들어온 뉴스가 과거 어떤 시장 상황의 뉴스 패턴과 가장 닮았는가?"를 7개 레이블(중심점)로 표현하고, **15거래일 뒤에 가격이 얼마나 움직일 가능성이 있는지**를 확률로 반환한다.

`run_market_news_training.py` 를 실행하면 XGBoost 비교 실험과 함께 자동으로 실행된다.

---

## 파일 구조

```
shared/
├── cluster/
│   ├── __init__.py     ← 공개 API 재export (model + visualize)
│   ├── model.py        ← 학습 · 추론 로직 전체
│   └── visualize.py    ← PCA 산점도 + 피처 히트맵 PNG 생성
├── pipelines/
│   └── market_news.py  ← step 7 에서 클러스터 학습 호출
└── run_market_news_training.py  ← 전체 실행 (XGBoost + 클러스터)
```

---

## 7개 레이블 정의

| 레이블 | 15일 선행 수익률 조건 | 의미 |
|---|---|---|
| `rise_3+` | +3% 이상 | 강한 상승 |
| `rise_2+` | +2% 이상 ~ +3% 미만 | 중간 상승 |
| `rise_1+` | +1% 이상 ~ +2% 미만 | 약한 상승 |
| `flat` | ±1% 미만 | 보합 (노이즈 레이블) |
| `fall_1+` | -1% 이하 ~ -2% 초과 | 약한 하락 |
| `fall_2+` | -2% 이하 ~ -3% 초과 | 중간 하락 |
| `fall_3+` | -3% 이하 | 강한 하락 |

`flat` 은 변동률이 작아 방향성 신호가 없는 날을 포착하는 노이즈 레이블이다.

---

## 로직 상세

### 학습 흐름 (`model.py`)

```
거래일 T
  │
  ├─ [feature] daily_news_features 에서 [T-15일, T) 구간 뉴스를 컬럼별 평균 → 13차원 벡터
  │
  └─ [label]   T 이후 15거래일 수익률 → 7개 구간 중 하나로 분류
```

**중심점 계산 방식 (label-conditioned prototype)**

레이블별로 평균 벡터를 독립적으로 계산한다. 순수 KMeans를 쓰면 데이터 분포에 따라 클러스터가 레이블 의미와 다르게 수렴할 수 있는 반면, 이 방식은 레이블 의미가 중심점에 직접 반영된다.

```
rise_3+ 중심점 = "15일 뒤 +3% 이상이었던 날들의 직전 15일 뉴스 평균"
flat    중심점 = "15일 뒤 ±1% 이하였던 날들의 직전 15일 뉴스 평균"
```

StandardScaler 로 전체 벡터를 정규화한 뒤 그룹별 평균을 구한다.

### 추론 흐름 (`model.py`)

```
새 뉴스 창 벡터 v (13차원)
  │
  ├─ StandardScaler 정규화 → v_scaled
  ├─ 7개 중심점 각각과 유클리디안 거리 계산  d_i = ||centroid_i - v_scaled||
  ├─ 역거리 비율로 확률 변환  p_i = (1/d_i) / Σ(1/d_j)
  └─ {label: probability} 딕셔너리 반환 (합 = 1.0)
```

### 시각화 (`visualize.py`)

- **좌측**: PCA 2D 산점도 — 15일 창 뉴스 벡터 전체(레이블별 색상) + 중심점(★)
- **우측**: 히트맵 — 7개 클러스터의 13개 피처 프로파일 (min-max 정규화 + 원값 표시)

### 벡터화에 사용하는 피처 (13개)

`daily_news_features.csv` 의 수치 컬럼만 사용한다. 날짜 주기 피처(`day_of_week_sin` 등)는 "언제 났는가"이지 "무슨 내용인가"가 아니므로 제외한다.

| 피처 | 설명 |
|---|---|
| `news_count` | 해당 날 뉴스 건수 |
| `category_BIS` | BIS 출처 비율 |
| `category_FOMC` | FOMC 출처 비율 |
| `category_UCSB` | UCSB(백악관) 출처 비율 |
| `title_positive_prob` | 제목 긍정 확률 (FinBERT) |
| `title_negative_prob` | 제목 부정 확률 |
| `title_neutral_prob` | 제목 중립 확률 |
| `title_sentiment_score` | 제목 감성 점수 (-1 ~ 1) |
| `body_positive_prob` | 본문 긍정 확률 |
| `body_negative_prob` | 본문 부정 확률 |
| `body_neutral_prob` | 본문 중립 확률 |
| `body_sentiment_score` | 본문 감성 점수 (-1 ~ 1) |
| `body_n_chunks` | 본문 분석 청크 수 (문서 길이 대리 지표) |

15일 창 안에 뉴스가 여러 날 있으면 날짜별 값을 **단순 평균**하여 창 전체를 하나의 벡터로 요약한다.

---

## 출력 파일

`python shared/run_market_news_training.py` 실행 시 자동 생성된다.

```
data/
└── training/
    └── comparison/
        ├── qqq_volatility_cluster_model.json   ← 추론에 필요한 수치 (centroids, scaler)
        ├── qqq_volatility_cluster_report.json  ← 클러스터 요약 (사람이 읽는 용)
        └── qqq_cluster_visualization.png       ← PCA 산점도 + 피처 히트맵
```

#### `cluster_model.json` 스키마

```json
{
  "centroids": [[...], ...],      // (7, 13) float 행렬 — VOLATILITY_LABELS 순서
  "scaler_mean": [...],           // StandardScaler.mean_ (13차원)
  "scaler_scale": [...],          // StandardScaler.scale_ (13차원)
  "feature_columns": ["news_count", ...],
  "labels": ["rise_3+", "rise_2+", "rise_1+", "flat", "fall_1+", "fall_2+", "fall_3+"],
  "horizon": 15,
  "window_days": 15
}
```

#### `cluster_report.json` 스키마

```json
{
  "clusters": [
    {
      "label": "rise_3+",
      "count": 142,
      "date_range": { "first": "2019-03-10", "last": "2025-11-20" },
      "centroid_feature_means": { "news_count": 1.23, "body_sentiment_score": 0.18, ... }
    },
    ...
  ]
}
```

---

## 실행 방법

### 전체 파이프라인 (권장)

```bash
python shared/run_market_news_training.py
```

XGBoost 비교 실험 완료 후 step 7 에서 클러스터 모델이 자동으로 학습된다.

### 시각화만 재생성

```bash
python shared/cluster/visualize.py
```

이미 저장된 `cluster_model.json` 을 읽어 PNG 를 다시 만든다.

---

## 추론 코드 예시

```python
import json
import pandas as pd
from shared.cluster import (
    load_cluster_model,
    predict_label_probabilities,
    get_closest_label,
)
from shared.cluster.model import _aggregate_news_window

# 1) 모델 로드
with open("data/training/comparison/qqq_volatility_cluster_model.json") as f:
    model_dict = json.load(f)

centroids, scaler = load_cluster_model(model_dict)

# 2) 최근 15일 뉴스 창 집계
news_df = pd.read_csv("data/crawler/features/daily_news_features.csv")
news_df["date"] = pd.to_datetime(news_df["date"])
news_indexed = news_df.set_index("date").sort_index()

today = pd.Timestamp.today().normalize()
vec = _aggregate_news_window(today, news_indexed, window_days=15)

if vec is not None:
    probs = predict_label_probabilities(vec, centroids, scaler)
    label, prob = get_closest_label(probs)
    print(f"예측: {label}  ({prob * 100:.1f}%)")
    for lbl, p in probs.items():
        print(f"  {lbl:8s} {p * 100:5.1f}%  {'█' * int(p * 40)}")
```

---

## 설계 결정 메모

**왜 KMeans가 아닌 레이블 조건부 평균인가**

KMeans는 뉴스 벡터의 자연 군집을 찾는다. 하지만 "상승 패턴 뉴스"와 "하락 패턴 뉴스"의 뉴스 공간 거리가 가깝다면 KMeans 클러스터는 가격 방향을 제대로 분리하지 못한다. 레이블 조건부 평균은 "실제로 그 결과가 나왔을 때의 뉴스가 평균적으로 어떻게 생겼는가"를 직접 추출하므로 클러스터 의미가 보장된다.

**왜 역거리 확률인가**

소프트맥스(-거리)도 흔히 쓰이지만, 역거리는 temperature 하이퍼파라미터가 없어 결과가 더 안정적이다. 여러 중심점이 비슷하게 멀면 확률이 고르게 퍼지고, 하나가 압도적으로 가까우면 그쪽 확률이 높아지는 직관적 동작을 한다.

**왜 달력일 15일 창인가 (거래일 아닌)**

뉴스는 주말/공휴일에도 발생한다. `daily_news_features` 는 이미 주말 뉴스를 다음 월요일로 이동시켜 저장하지만, 창 자체는 달력 기준으로 끊어야 "최근 약 3주"의 뉴스를 일관되게 포함할 수 있다.

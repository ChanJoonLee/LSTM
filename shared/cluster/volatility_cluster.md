# 뉴스 기반 변동률 클러스터 모델

## 개요

`shared/cluster/` 패키지에 구현된 모듈이다.

"지금 들어온 뉴스가 과거 어떤 시장 상황의 뉴스 패턴과 가장 닮았는가?"를 5개 레이블(중심점)로 표현하고, **7거래일 뒤에 가격이 얼마나 움직일 가능성이 있는지**를 확률로 반환한다.

`run_market_news_training.py` 를 실행하면 XGBoost 비교 실험과 함께 자동으로 실행된다.

---

## 파일 구조

```
shared/
├── cluster/
│   ├── __init__.py            ← 공개 API 재export (model + visualize)
│   ├── model.py               ← 학습 · 추론 로직 전체
│   ├── visualize.py           ← PCA 산점도 + 피처 히트맵 PNG 생성
│   └── volatility_cluster.md  ← 이 문서
├── pipelines/
│   └── market_news.py         ← step 7 에서 클러스터 학습 호출
└── run_market_news_training.py ← 전체 실행 (XGBoost + 클러스터)
```

---

## 5개 레이블 정의

레이블 경계는 **고정 수치가 아니라 학습 데이터의 분위수**로 결정된다. 전체 7일 선행 수익률 분포의 20/40/60/80 백분위를 경계로 삼아 각 레이블에 약 20%씩 균등하게 배분한다.

| 레이블 | 수익률 구간 | 의미 |
|---|---|---|
| `rise_strong` | 상위 20% (> q80) | 강한 상승 |
| `rise` | 60~80 백분위 | 상승 |
| `flat` | 40~60 백분위 | 보합 |
| `fall` | 20~40 백분위 | 하락 |
| `fall_strong` | 하위 20% (< q20) | 강한 하락 |

> **분위수 경계를 쓰는 이유**  
> 고정 ±1%/2%/3% 경계는 QQQ처럼 상승 편향이 있는 자산에서 레이블 간 샘플 수가 크게 불균형해진다 (예: `rise_3+` 626개, `fall_2+` 46개). 샘플이 적은 레이블은 중심점 신뢰도가 낮아진다. 분위수 경계는 모든 레이블에 동등한 샘플을 보장해 중심점을 안정적으로 만든다.
>
> 실제 학습된 경계값 예시 (QQQ 2019–2025):
> ```
> q20 = -2.6%  q40 = +1.2%  q60 = +3.1%  q80 = +5.0%
> ```
> `flat` 이 +1.2% ~ +3.1% 에 걸치는 것은 QQQ의 상승 편향을 반영한 자연스러운 결과다.

---

## 로직 상세

### 학습 흐름 (`model.py`)

```
거래일 T
  │
  ├─ [feature] daily_news_features 에서 [T-7일, T) 구간 뉴스를 컬럼별 평균 → 10차원 벡터
  │
  └─ [label]   T 이후 7거래일 수익률 → 분위수 경계로 5개 구간 중 하나로 분류
```

**중심점 계산 방식 (label-conditioned prototype)**

레이블별로 평균 벡터를 독립적으로 계산한다. 순수 KMeans를 쓰면 데이터 분포에 따라 클러스터가 레이블 의미와 다르게 수렴할 수 있는 반면, 이 방식은 레이블 의미가 중심점에 직접 반영된다.

```
rise_strong 중심점 = "7일 뒤 상위 20% 수익률이었던 날들의 직전 7일 뉴스 평균"
flat        중심점 = "7일 뒤 40~60 백분위 수익률이었던 날들의 직전 7일 뉴스 평균"
```

StandardScaler 로 전체 벡터를 정규화한 뒤 그룹별 평균을 구한다.

### 추론 흐름 (`model.py`)

```
새 뉴스 창 벡터 v (10차원)
  │
  ├─ StandardScaler 정규화 → v_scaled
  ├─ 5개 중심점 각각과 유클리디안 거리 계산  d_i = ||centroid_i - v_scaled||
  ├─ 역거리 비율로 확률 변환  p_i = (1/d_i) / Σ(1/d_j)
  └─ {label: probability} 딕셔너리 반환 (합 = 1.0)
```

### 시각화 (`visualize.py`)

- **좌측**: PCA 2D 산점도 — 7일 창 뉴스 벡터 전체(레이블별 색상) + 중심점(★)
- **우측**: 히트맵 — 5개 클러스터의 10개 피처 프로파일 (min-max 정규화 + 원값 표시)

### 벡터화에 사용하는 피처 (10개)

`daily_news_features.csv` 의 수치 컬럼만 사용한다. 날짜 주기 피처(`day_of_week_sin` 등)는 "언제 났는가"이지 "무슨 내용인가"가 아니므로 제외한다.

아래 피처는 **scaled 중심점 간 표준편차** 기준으로 분리도가 낮아 제거됐다: `category_BIS` (0.068), `body_sentiment_score` (0.067), `body_n_chunks` (항상 0).

| 피처 | 설명 | scaled std |
|---|---|---|
| `body_neutral_prob` | 본문 중립 확률 | 0.146 |
| `title_sentiment_score` | 제목 감성 점수 | 0.145 |
| `body_positive_prob` | 본문 긍정 확률 | 0.129 |
| `title_positive_prob` | 제목 긍정 확률 (FinBERT) | 0.115 |
| `title_negative_prob` | 제목 부정 확률 | 0.112 |
| `category_FOMC` | FOMC 출처 비율 | 0.083 |
| `category_UCSB` | UCSB(백악관) 출처 비율 | 0.080 |
| `title_neutral_prob` | 제목 중립 확률 | 0.075 |
| `news_count` | 해당 날 뉴스 건수 | 0.073 |
| `body_negative_prob` | 본문 부정 확률 | 0.072 |

7일 창 안에 뉴스가 여러 날 있으면 날짜별 값을 **단순 평균**하여 창 전체를 하나의 벡터로 요약한다.

---

## 출력 파일

`python shared/run_market_news_training.py` 실행 시 자동 생성된다.

```
data/
└── training/
    └── comparison/
        ├── qqq_volatility_cluster_model.json   ← 추론에 필요한 수치 (centroids, scaler, thresholds)
        ├── qqq_volatility_cluster_report.json  ← 클러스터 요약 (사람이 읽는 용)
        └── qqq_cluster_visualization.png       ← PCA 산점도 + 피처 히트맵
```

#### `cluster_model.json` 스키마

```json
{
  "centroids": [[...], ...],           // (5, 10) float 행렬 — VOLATILITY_LABELS 순서
  "scaler_mean": [...],                // StandardScaler.mean_ (10차원)
  "scaler_scale": [...],               // StandardScaler.scale_ (10차원)
  "feature_columns": ["news_count", ...],
  "labels": ["rise_strong", "rise", "flat", "fall", "fall_strong"],
  "quantile_thresholds": [-0.026, 0.012, 0.031, 0.050],  // [q20, q40, q60, q80]
  "horizon": 15,
  "window_days": 15
}
```

`quantile_thresholds` 는 추론 시 레이블 경계 재현에 필요하지 않지만 (추론은 중심점 거리만 사용), 학습 데이터의 수익률 분포를 문서화하는 용도로 저장한다.

#### `cluster_report.json` 스키마

```json
{
  "clusters": [
    {
      "label": "rise_strong",
      "count": 330,
      "date_range": { "first": "2019-03-27", "last": "2025-11-18" },
      "centroid_feature_means": { "news_count": 1.57, "body_neutral_prob": 0.10, ... }
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

centroids, scaler, thresholds = load_cluster_model(model_dict)

# 2) 최근 7일 뉴스 창 집계
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
        print(f"  {lbl:12s} {p * 100:5.1f}%  {'█' * int(p * 40)}")
```

---

## 설계 결정 메모

**왜 KMeans가 아닌 레이블 조건부 평균인가**

KMeans는 뉴스 벡터의 자연 군집을 찾는다. 하지만 "상승 패턴 뉴스"와 "하락 패턴 뉴스"의 뉴스 공간 거리가 가깝다면 KMeans 클러스터는 가격 방향을 제대로 분리하지 못한다. 레이블 조건부 평균은 "실제로 그 결과가 나왔을 때의 뉴스가 평균적으로 어떻게 생겼는가"를 직접 추출하므로 클러스터 의미가 보장된다.

**왜 역거리 확률인가**

소프트맥스(-거리)도 흔히 쓰이지만, 역거리는 temperature 하이퍼파라미터가 없어 결과가 더 안정적이다. 여러 중심점이 비슷하게 멀면 확률이 고르게 퍼지고, 하나가 압도적으로 가까우면 그쪽 확률이 높아지는 직관적 동작을 한다.

**왜 달력일 7일 창인가 (거래일 아닌)**

뉴스는 주말/공휴일에도 발생한다. `daily_news_features` 는 이미 주말 뉴스를 다음 월요일로 이동시켜 저장하지만, 창 자체는 달력 기준으로 끊어야 "최근 약 3주"의 뉴스를 일관되게 포함할 수 있다.

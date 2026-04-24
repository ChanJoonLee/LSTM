# 뉴스 기반 변동률 클러스터 모델

## 개요

`shared/news/volatility_cluster.py`에 구현된 모듈이다.

"지금 들어온 뉴스가 과거 어떤 시장 상황의 뉴스 패턴과 가장 닮았는가?"를 7개 레이블(중심점)로 표현하고, **15거래일 뒤에 가격이 얼마나 움직일 가능성이 있는지**를 확률로 반환한다.

---

## 7개 레이블 정의

| 레이블 | 15일 선행 수익률 조건 | 의미 |
|---|---|---|
| `rise_3+` | +3% 이상 | 강한 상승 |
| `rise_2+` | +2% 이상 ~ +3% 미만 | 중간 상승 |
| `rise_1+` | +1% 이상 ~ +2% 미만 | 약한 상승 |
| `flat` | ±1% 미만 | 보합 (노이즈) |
| `fall_1+` | -1% 이하 ~ -2% 초과 | 약한 하락 |
| `fall_2+` | -2% 이하 ~ -3% 초과 | 중간 하락 |
| `fall_3+` | -3% 이하 | 강한 하락 |

`flat`은 변동률이 작아 의미있는 신호가 없는 날을 포착하는 "노이즈 레이블"이다. 새 뉴스가 이 중심점에 가깝다면 뚜렷한 방향성 신호가 없다는 뜻이다.

---

## 로직 상세

### 학습 흐름

```
거래일 T
  │
  ├─ [feature] daily_news_features에서 [T-15일, T) 구간 뉴스를 컬럼별 평균 → 13차원 벡터
  │
  └─ [label]   T+15 거래일 후 수익률 = (price[T+15] - price[T]) / price[T]
                → 7개 구간 중 하나로 분류
```

전체 거래일 중 해당 구간에 뉴스가 하나라도 있는 날만 학습 샘플로 포함된다.

**중심점 계산 방식 (label-conditioned prototype)**

순수 KMeans 대신 레이블별 평균을 중심점으로 사용한다. KMeans는 데이터 분포에 따라 클러스터가 의도한 레이블과 다르게 수렴할 수 있는 반면, 이 방식은 레이블 의미가 중심점에 직접 반영된다.

```
rise_3+ 중심점 = "15일 뒤 +3% 이상이었던 날들의 직전 15일 뉴스 평균"
flat    중심점 = "15일 뒤 ±1% 이하였던 날들의 직전 15일 뉴스 평균"
...
```

StandardScaler로 전체 학습 벡터를 정규화한 뒤 그룹별 평균을 구한다. 스케일링을 하지 않으면 `news_count` 같은 절대 수치가 감성 확률값(0~1)을 압도하는 문제가 생긴다.

### 추론 흐름

```
새 뉴스 창 벡터 v (13차원)
  │
  ├─ StandardScaler로 정규화 → v_scaled
  │
  ├─ 7개 중심점 각각과 유클리디안 거리 계산
  │     d_i = ||centroid_i - v_scaled||
  │
  ├─ 역거리 비율로 확률 변환
  │     p_i = (1/d_i) / Σ(1/d_j)
  │
  └─ {label: probability} 딕셔너리 반환
```

거리가 작을수록(= 중심점에 가까울수록) 확률이 높아진다. 확률의 합은 항상 1.0이다.

### 벡터화에 사용하는 피처 (13개)

`daily_news_features.csv`의 수치 컬럼을 그대로 사용한다. 날짜 주기 피처(`day_of_week_sin` 등)는 "언제 났는가"이지 "무슨 내용인가"가 아니므로 제외한다.

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

## 파일 구조

### 모듈 위치

```
shared/
└── news/
    └── volatility_cluster.py   ← 이 모듈
```

### 출력 파일

파이프라인(`python shared/run_market_news_training.py`) 실행 시 자동 생성된다.

```
data/
└── training/
    └── comparison/
        ├── qqq_volatility_cluster_model.json   ← 추론에 필요한 수치 (centroids, scaler)
        └── qqq_volatility_cluster_report.json  ← 사람이 읽는 클러스터 요약
```

#### `cluster_model.json` 스키마

```json
{
  "centroids": [[...], [...], ...],   // (7, 13) float 행렬 — VOLATILITY_LABELS 순서
  "scaler_mean": [...],               // StandardScaler.mean_ (13차원)
  "scaler_scale": [...],              // StandardScaler.scale_ (13차원)
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
      "date_range": { "first": "2015-03-10", "last": "2025-11-20" },
      "centroid_feature_means": {
        "news_count": 1.23,
        "body_sentiment_score": 0.18,
        ...
      }
    },
    ...
  ]
}
```

---

## 파이프라인 연결 지점

`shared/pipelines/market_news.py`의 `run_market_news_training_pipeline()` 7번 단계에서 실행된다. XGBoost 실험이 모두 끝난 뒤 자동으로 돌아간다.

```
1. 뉴스 원본 로드
2. 일자별 뉴스 피처 생성      ← daily_news_features 생성
3. 시장 피처 생성             ← market_feature_df 생성
4. market_only 실험
5. market+news 실험
6. 정렬 비교 실험
7. [클러스터 모델 학습]       ← 2, 3의 결과물을 그대로 재사용
```

---

## 추론 사용법

### 기본 예시

```python
import json
import pandas as pd
from shared.news.volatility_cluster import (
    load_cluster_model,
    predict_label_probabilities,
    get_closest_label,
    _aggregate_news_window,
    CLUSTER_FEATURE_COLS,
)

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

if vec is None:
    print("최근 15일 이내 뉴스 없음")
else:
    # 3) 확률 예측
    probs = predict_label_probabilities(vec, centroids, scaler)
    label, prob = get_closest_label(probs)

    print(f"가장 유사한 패턴: {label}  ({prob * 100:.1f}%)")
    print("전체 확률 분포:")
    for lbl, p in probs.items():
        bar = "█" * int(p * 40)
        print(f"  {lbl:8s} {p * 100:5.1f}%  {bar}")
```

### 출력 예시

```
가장 유사한 패턴: fall_1+  (38.2%)
전체 확률 분포:
  rise_3+    4.1%  █
  rise_2+    6.3%  ██
  rise_1+   11.2%  ████
  flat      22.5%  █████████
  fall_1+   38.2%  ███████████████
  fall_2+   12.1%  ████
  fall_3+    5.6%  ██
```

### 특정 날짜로 소급 테스트

```python
target_date = pd.Timestamp("2025-01-15")
vec = _aggregate_news_window(target_date, news_indexed, window_days=15)
probs = predict_label_probabilities(vec, centroids, scaler)
label, prob = get_closest_label(probs)
print(f"{target_date.date()} 기준 → {label} ({prob * 100:.1f}%)")
```

---

## 설계 결정 메모

**왜 KMeans가 아닌 레이블 조건부 평균인가**

KMeans는 뉴스 벡터의 자연 군집을 찾는다. 하지만 "상승 패턴 뉴스"와 "하락 패턴 뉴스"의 뉴스 공간 거리가 가깝다면 KMeans 클러스터는 가격 방향을 제대로 분리하지 못한다. 레이블 조건부 평균은 "실제로 그 결과가 나왔을 때의 뉴스가 평균적으로 어떻게 생겼는가"를 직접 추출하므로, 클러스터 의미가 보장된다.

**왜 역거리 확률인가**

소프트맥스(-거리)도 흔히 쓰이지만, 역거리는 하이퍼파라미터(temperature)가 없어 결과가 더 안정적이다. 여러 중심점이 비슷하게 멀다면 확률이 고르게 퍼지고, 하나가 압도적으로 가깝다면 그쪽 확률이 크게 높아지는 직관적 동작을 한다.

**왜 달력일 15일 창인가 (거래일 아닌)**

뉴스는 주말/공휴일에도 발생한다. `daily_news_features`는 이미 주말 뉴스를 다음 월요일로 이동시켜 저장하지만, 창 자체는 달력 기준으로 끊어야 "최근 약 3주"의 뉴스를 일관되게 포함할 수 있다.

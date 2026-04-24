from __future__ import annotations

"""
뉴스 패턴과 15일 선행 가격 변동률의 관계를 7개 레이블 중심점으로 모델링한다.

학습 흐름:
  1. 각 거래일의 15일 선행 수익률로 레이블을 붙인다.
  2. anchor 날짜 직전 15 달력일의 daily_news_features 를 평균 벡터로 집계한다.
  3. 레이블별 평균 벡터를 중심점으로 확정한다 (label-conditioned prototype).

추론 흐름:
  - 최근 15일 뉴스 창을 같은 방식으로 집계한다.
  - 7개 중심점까지의 역거리를 확률로 변환해 반환한다.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURE_COLS: list[str] = [
    "news_count",
    "category_BIS",
    "category_FOMC",
    "category_UCSB",
    "title_positive_prob",
    "title_negative_prob",
    "title_neutral_prob",
    "title_sentiment_score",
    "body_positive_prob",
    "body_negative_prob",
    "body_neutral_prob",
    "body_sentiment_score",
    "body_n_chunks",
]

VOLATILITY_LABELS: list[str] = [
    "rise_3+",
    "rise_2+",
    "rise_1+",
    "flat",
    "fall_1+",
    "fall_2+",
    "fall_3+",
]

# (label, lower_inclusive, upper_exclusive) — 15일 선행 수익률 기준
_LABEL_BOUNDS: list[tuple[str, float, float]] = [
    ("rise_3+",  0.03,          float("inf")),
    ("rise_2+",  0.02,          0.03),
    ("rise_1+",  0.01,          0.02),
    ("flat",    -0.01,          0.01),
    ("fall_1+", -0.02,         -0.01),
    ("fall_2+", -0.03,         -0.02),
    ("fall_3+", float("-inf"), -0.03),
]


def _assign_label(ret: float) -> str:
    for label, low, high in _LABEL_BOUNDS:
        if low <= ret < high:
            return label
    return "flat"


def _aggregate_news_window(
    anchor_date: pd.Timestamp,
    news_indexed: pd.DataFrame,
    window_days: int,
) -> np.ndarray | None:
    """anchor_date 직전 window_days 달력일의 뉴스 피처를 평균 벡터로 집계한다.

    news_indexed 는 date 를 인덱스로 가진 DataFrame 이어야 한다.
    """
    cutoff = anchor_date - pd.Timedelta(days=window_days)
    upper = anchor_date - pd.Timedelta(days=1)
    try:
        window = news_indexed.loc[cutoff:upper, CLUSTER_FEATURE_COLS]
    except KeyError:
        return None
    if window.empty:
        return None
    return window.mean(axis=0).to_numpy(dtype=float)


def build_event_dataset(
    market_df: pd.DataFrame,
    daily_news_df: pd.DataFrame,
    horizon: int = 15,
    window_days: int = 15,
) -> tuple[np.ndarray, list[str], list[pd.Timestamp]]:
    """각 거래일의 horizon-일 선행 수익률로 레이블을 붙이고 뉴스 창 벡터를 반환한다.

    Parameters
    ----------
    market_df     : Date, target_price 컬럼을 포함한 시장 피처 프레임
    daily_news_df : date 컬럼을 포함한 일자별 뉴스 피처 테이블
    horizon       : 선행 수익률 계산 거래일 수
    window_days   : anchor 이전 달력일 수 (뉴스 창 크기)

    Returns
    -------
    vectors : (N, n_features) float 배열
    labels  : N 길이 레이블 리스트 (VOLATILITY_LABELS 중 하나)
    dates   : N 길이 anchor 날짜 리스트
    """
    df = market_df[["Date", "target_price"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret_fwd"] = df["target_price"].pct_change(horizon).shift(-horizon)

    news = daily_news_df.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.tz_localize(None)
    news = news.sort_values("date")
    news_indexed = news.set_index("date").sort_index()

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    dates: list[pd.Timestamp] = []

    for _, row in df.iterrows():
        ret = row["ret_fwd"]
        if pd.isna(ret):
            continue
        vec = _aggregate_news_window(row["Date"], news_indexed, window_days)
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(_assign_label(float(ret)))
        dates.append(row["Date"])

    return np.array(vectors, dtype=float), labels, dates


def fit_news_centroids(
    vectors: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """각 레이블 그룹의 평균 스케일 벡터를 중심점으로 계산한다.

    Returns
    -------
    centroids : (7, n_features) — VOLATILITY_LABELS 순서
    counts    : (7,) — 각 그룹 샘플 수
    scaler    : 전체 데이터로 fit 된 StandardScaler
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    n_features = scaled.shape[1]
    centroids = np.zeros((len(VOLATILITY_LABELS), n_features))
    counts = np.zeros(len(VOLATILITY_LABELS), dtype=int)

    label_index = {label: i for i, label in enumerate(VOLATILITY_LABELS)}
    for i, label in enumerate(VOLATILITY_LABELS):
        idx = [j for j, l in enumerate(labels) if l == label]
        counts[i] = len(idx)
        if idx:
            centroids[i] = scaled[np.array(idx)].mean(axis=0)

    return centroids, counts, scaler


def predict_label_probabilities(
    news_window_vector: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, float]:
    """새 뉴스 창 벡터와 7개 중심점 간 역거리 비율을 확률로 반환한다.

    Parameters
    ----------
    news_window_vector : (n_features,) 원래 스케일 벡터
    """
    scaled = scaler.transform(news_window_vector.reshape(1, -1))[0]
    distances = np.linalg.norm(centroids - scaled, axis=1)
    inv_dist = 1.0 / (distances + 1e-9)
    probs = inv_dist / inv_dist.sum()
    return {label: float(probs[i]) for i, label in enumerate(VOLATILITY_LABELS)}


def get_closest_label(probabilities: dict[str, float]) -> tuple[str, float]:
    """확률 딕셔너리에서 최고 확률 레이블과 그 확률을 반환한다."""
    best = max(probabilities, key=probabilities.__getitem__)
    return best, probabilities[best]


def build_cluster_summary(
    labels: list[str],
    dates: list[pd.Timestamp],
    centroids: np.ndarray,
    counts: np.ndarray,
    scaler: StandardScaler,
) -> list[dict]:
    """각 클러스터(레이블)의 샘플 수, 날짜 범위, 중심점 피처 평균을 반환한다."""
    summary = []
    for i, label in enumerate(VOLATILITY_LABELS):
        idx = [j for j, l in enumerate(labels) if l == label]
        if idx:
            selected = [dates[j] for j in idx]
            date_range: dict = {
                "first": str(min(selected).date()),
                "last": str(max(selected).date()),
            }
        else:
            date_range = {}

        centroid_original = scaler.inverse_transform(centroids[i].reshape(1, -1))[0]
        summary.append(
            {
                "label": label,
                "count": int(counts[i]),
                "date_range": date_range,
                "centroid_feature_means": {
                    col: round(float(centroid_original[j]), 4)
                    for j, col in enumerate(CLUSTER_FEATURE_COLS)
                },
            }
        )
    return summary


def load_cluster_model(model_dict: dict) -> tuple[np.ndarray, StandardScaler]:
    """write_json 으로 저장된 클러스터 모델 딕셔너리에서 centroids 와 scaler 를 복원한다."""
    centroids = np.array(model_dict["centroids"], dtype=float)
    scaler = StandardScaler()
    scaler.mean_ = np.array(model_dict["scaler_mean"], dtype=float)
    scaler.scale_ = np.array(model_dict["scaler_scale"], dtype=float)
    scaler.n_features_in_ = len(scaler.mean_)
    return centroids, scaler

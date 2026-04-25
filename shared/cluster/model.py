from __future__ import annotations

"""
뉴스 패턴과 7일 선행 가격 변동률의 관계를 5개 레이블 중심점으로 모델링한다.

학습 흐름:
  1. 각 거래일의 7일 선행 수익률을 모아 20/40/60/80 백분위 경계를 계산한다.
  2. 백분위 경계로 5개 레이블을 붙인다 (fall_strong / fall / flat / rise / rise_strong).
  3. anchor 날짜 직전 7 달력일의 daily_news_features 를 평균 벡터로 집계한다.
  4. 레이블별 평균 벡터를 중심점으로 확정한다 (label-conditioned prototype).

추론 흐름:
  - 최근 7일 뉴스 창을 같은 방식으로 집계한다.
  - 5개 중심점까지의 역거리를 확률로 변환해 반환한다.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURE_COLS: list[str] = [
    "news_count",
    "category_FOMC",
    "category_UCSB",
    "title_positive_prob",
    "title_negative_prob",
    "title_neutral_prob",
    "title_sentiment_score",
    "body_positive_prob",
    "body_negative_prob",
    "body_neutral_prob",
]

VOLATILITY_LABELS: list[str] = [
    "rise_strong",
    "rise",
    "flat",
    "fall",
    "fall_strong",
]


class QuantileThresholds(NamedTuple):
    q20: float
    q40: float
    q60: float
    q80: float


def compute_quantile_thresholds(returns: np.ndarray) -> QuantileThresholds:
    """학습 수익률 분포의 20/40/60/80 백분위를 레이블 경계로 반환한다."""
    q20, q40, q60, q80 = np.quantile(returns, [0.20, 0.40, 0.60, 0.80])
    return QuantileThresholds(float(q20), float(q40), float(q60), float(q80))


def _assign_label_quantile(ret: float, thresholds: QuantileThresholds) -> str:
    if ret >= thresholds.q80:
        return "rise_strong"
    if ret >= thresholds.q60:
        return "rise"
    if ret >= thresholds.q40:
        return "flat"
    if ret >= thresholds.q20:
        return "fall"
    return "fall_strong"


def _label_indices(labels: list[str], target: str) -> list[int]:
    return [i for i, l in enumerate(labels) if l == target]


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
) -> tuple[np.ndarray, list[str], list[pd.Timestamp], QuantileThresholds]:
    """각 거래일의 horizon-일 선행 수익률로 레이블을 붙이고 뉴스 창 벡터를 반환한다.

    레이블 경계는 학습 수익률의 20/40/60/80 백분위로 자동 결정된다.

    Parameters
    ----------
    market_df     : Date, target_price 컬럼을 포함한 시장 피처 프레임
    daily_news_df : date 컬럼을 포함한 일자별 뉴스 피처 테이블
    horizon       : 선행 수익률 계산 거래일 수 (기본 7)
    window_days   : anchor 이전 달력일 수 (뉴스 창 크기, 기본 7)

    Returns
    -------
    vectors    : (N, n_features) float 배열
    labels     : N 길이 레이블 리스트 (VOLATILITY_LABELS 중 하나)
    dates      : N 길이 anchor 날짜 리스트
    thresholds : 추론 시 동일 경계 재현을 위해 모델과 함께 저장해야 한다
    """
    df = market_df[["Date", "target_price"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df["ret_fwd"] = df["target_price"].pct_change(horizon).shift(-horizon)

    news = daily_news_df.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.tz_localize(None)
    news_indexed = news.sort_values("date").set_index("date").sort_index()

    valid_returns = df["ret_fwd"].dropna().to_numpy(dtype=float)
    thresholds = compute_quantile_thresholds(valid_returns)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    dates: list[pd.Timestamp] = []

    for row in df.itertuples(index=False):
        ret = row.ret_fwd
        if pd.isna(ret):
            continue
        vec = _aggregate_news_window(row.Date, news_indexed, window_days)
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(_assign_label_quantile(float(ret), thresholds))
        dates.append(row.Date)

    return np.array(vectors, dtype=float), labels, dates, thresholds


def fit_news_centroids(
    vectors: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """각 레이블 그룹의 평균 스케일 벡터를 중심점으로 계산한다.

    Returns
    -------
    centroids : (5, n_features) — VOLATILITY_LABELS 순서
    counts    : (5,) — 각 그룹 샘플 수
    scaler    : 전체 데이터로 fit 된 StandardScaler
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    n_features = scaled.shape[1]
    centroids = np.zeros((len(VOLATILITY_LABELS), n_features))
    counts = np.zeros(len(VOLATILITY_LABELS), dtype=int)

    for i, label in enumerate(VOLATILITY_LABELS):
        idx = _label_indices(labels, label)
        counts[i] = len(idx)
        if idx:
            centroids[i] = scaled[np.array(idx)].mean(axis=0)

    return centroids, counts, scaler


def predict_label_probabilities(
    news_window_vector: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, float]:
    """새 뉴스 창 벡터와 5개 중심점 간 역거리 비율을 확률로 반환한다.

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
        idx = _label_indices(labels, label)
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


def load_cluster_model(
    model_dict: dict,
) -> tuple[np.ndarray, StandardScaler, QuantileThresholds]:
    """write_json 으로 저장된 클러스터 모델 딕셔너리에서 centroids, scaler, thresholds 를 복원한다."""
    centroids = np.array(model_dict["centroids"], dtype=float)
    scaler = StandardScaler()
    scaler.mean_ = np.array(model_dict["scaler_mean"], dtype=float)
    scaler.scale_ = np.array(model_dict["scaler_scale"], dtype=float)
    scaler.n_features_in_ = len(scaler.mean_)
    thresholds = QuantileThresholds(*model_dict["quantile_thresholds"])
    return centroids, scaler, thresholds

from __future__ import annotations

"""
뉴스 클러스터 파이프라인.

daily_news_features + 시장 가격 데이터로 7개 변동률 레이블 중심점 모델을 학습한다.
XGBoost 비교 실험(market_news 파이프라인)과 완전히 독립적으로 실행할 수 있다.

실행 순서:
  1. daily_news_features CSV 로드 (크롤러 산출물)
  2. 시장 가격 데이터 다운로드 → target_price 추출
  3. 15일 창 이벤트 데이터셋 생성 (벡터 + 레이블)
  4. 레이블 조건부 중심점 학습
  5. 모델 JSON / 리포트 JSON / PCA 시각화 PNG 저장
"""

import pandas as pd

from shared.common.utils import write_json
from shared.config.schema import VolatilityClusterConfig
from shared.market.data import build_market_feature_frame, download_market_data
from shared.news.visualize_clusters import save_cluster_visualization
from shared.news.volatility_cluster import (
    CLUSTER_FEATURE_COLS,
    VOLATILITY_LABELS,
    build_cluster_summary,
    build_event_dataset,
    fit_news_centroids,
)

__all__ = ["run_volatility_cluster_pipeline"]


def run_volatility_cluster_pipeline(config: VolatilityClusterConfig) -> dict:
    """
    뉴스 클러스터 모델을 학습하고 결과를 저장한다.

    Returns
    -------
    dict : cluster_report_payload  {"clusters": [...]}
    """
    # 1) 크롤러가 생성한 daily_news_features 를 그대로 로드한다.
    daily_news = pd.read_csv(
        config.daily_news_features_input_path,
        encoding="utf-8-sig",
    )

    # 2) target_price 시계열을 얻기 위해 시장 데이터를 다운로드한다.
    #    클러스터 파이프라인은 XGBoost 피처 전체가 아닌 가격 컬럼만 필요하다.
    raw_market = download_market_data(config)
    market_df, _ = build_market_feature_frame(raw_market, config.target_ticker)

    # 3) 각 거래일의 15일 선행 수익률로 레이블을 붙이고 뉴스 창 벡터를 생성한다.
    vectors, labels, dates = build_event_dataset(
        market_df,
        daily_news,
        horizon=config.cluster_horizon,
        window_days=config.cluster_window_days,
    )

    # 4) 레이블 조건부 평균으로 7개 중심점을 계산한다.
    centroids, counts, scaler = fit_news_centroids(vectors, labels)
    cluster_summary = build_cluster_summary(labels, dates, centroids, counts, scaler)

    # 5) 모델 / 리포트 / 시각화를 저장한다.
    cluster_model_payload: dict = {
        "centroids": centroids.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_columns": CLUSTER_FEATURE_COLS,
        "labels": VOLATILITY_LABELS,
        "horizon": config.cluster_horizon,
        "window_days": config.cluster_window_days,
    }
    write_json(cluster_model_payload, config.cluster_model_output_path)

    cluster_report_payload: dict = {"clusters": cluster_summary}
    write_json(cluster_report_payload, config.cluster_report_output_path)

    save_cluster_visualization(
        vectors=vectors,
        labels=labels,
        counts=counts,
        centroids=centroids,
        scaler=scaler,
        output_path=config.cluster_visualization_output_path,
        horizon=config.cluster_horizon,
        window_days=config.cluster_window_days,
    )

    return cluster_report_payload

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from shared.config.schema import VolatilityClusterConfig
from shared.pipelines.volatility_cluster import run_volatility_cluster_pipeline


def parse_args() -> argparse.Namespace:
    default = VolatilityClusterConfig()
    parser = argparse.ArgumentParser(
        description="뉴스 클러스터 모델 학습 (XGBoost 실험 없이 단독 실행)"
    )
    parser.add_argument("--target-ticker", default=default.target_ticker)
    parser.add_argument("--start-date", default=default.start_date)
    parser.add_argument("--end-date", default=default.end_date)
    parser.add_argument(
        "--daily-news-input",
        default=str(default.daily_news_features_input_path),
    )
    parser.add_argument(
        "--cluster-model-output",
        default=str(default.cluster_model_output_path),
    )
    parser.add_argument(
        "--cluster-report-output",
        default=str(default.cluster_report_output_path),
    )
    parser.add_argument(
        "--cluster-viz-output",
        default=str(default.cluster_visualization_output_path),
    )
    parser.add_argument("--horizon", type=int, default=default.cluster_horizon)
    parser.add_argument("--window-days", type=int, default=default.cluster_window_days)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VolatilityClusterConfig(
        target_ticker=args.target_ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        daily_news_features_input_path=Path(args.daily_news_input),
        cluster_model_output_path=Path(args.cluster_model_output),
        cluster_report_output_path=Path(args.cluster_report_output),
        cluster_visualization_output_path=Path(args.cluster_viz_output),
        cluster_horizon=args.horizon,
        cluster_window_days=args.window_days,
    )

    result = run_volatility_cluster_pipeline(config)
    clusters = result.get("clusters", [])

    print("=== Volatility Cluster Training Completed ===")
    print(f"Target ticker : {config.target_ticker}")
    print(f"Horizon       : {config.cluster_horizon}d  |  Window: {config.cluster_window_days}d")
    print("--- Clusters ---")
    for cluster in clusters:
        label = cluster["label"]
        count = cluster["count"]
        dr = cluster.get("date_range", {})
        date_str = f"{dr.get('first', '?')} ~ {dr.get('last', '?')}" if dr else "N/A"
        print(f"  [{label:8s}]  n={count:4d}  ({date_str})")
    print(f"Model saved to:         {config.cluster_model_output_path}")
    print(f"Report saved to:        {config.cluster_report_output_path}")
    print(f"Visualization saved to: {config.cluster_visualization_output_path}")


if __name__ == "__main__":
    main()

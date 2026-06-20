from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from shared.config.schema import make_training_config
from shared.config.ticker_presets import available_ticker_presets
from shared.pipelines.market_news import run_market_news_training_pipeline  # noqa: E402


REPORT_WIDTH = 96


def _parse_horizon_candidates(raw_value: str) -> tuple[int, ...]:
    """
    CLI 입력 문자열을 horizon 튜플로 바꾼다.

    예:
    - "5,10,15" -> (5, 10, 15)
    """
    horizon_values = []

    for value in raw_value.split(","):
        stripped = value.strip()
        if not stripped:
            continue
        horizon_values.append(int(stripped))

    if not horizon_values:
        raise ValueError("At least one horizon value is required.")

    return tuple(horizon_values)


def _parse_csv_values(raw_value: str) -> tuple[str, ...]:
    values = tuple(
        value.strip().upper()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not values:
        raise ValueError("At least one comma-separated value is required.")
    return values


def _parse_macro_tickers(raw_value: str) -> tuple[str, ...]:
    return _parse_csv_values(raw_value)


def _parse_market_feature_columns(raw_value: str) -> tuple[str, ...]:
    columns = tuple(
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not columns:
        raise ValueError("At least one market feature column is required.")
    return columns


def _parse_supplementary_ticker_feature_suffixes(raw_value: str) -> tuple[str, ...]:
    if raw_value.strip().lower() in {"", "none"}:
        return ()
    return _parse_market_feature_columns(raw_value)


def _optional_path_overrides(args: argparse.Namespace) -> dict:
    output_arg_to_config_field = {
        "market_only_training_output": "market_only_training_frame_output_path",
        "market_only_predictions_output": "market_only_predictions_output_path",
        "market_only_model_output": "market_only_model_output_path",
        "market_only_metadata_output": "market_only_metadata_output_path",
        "daily_news_output": "daily_news_features_output_path",
        "merged_training_output": "merged_training_frame_output_path",
        "predictions_output": "predictions_output_path",
        "model_output": "model_output_path",
        "metadata_output": "metadata_output_path",
        "comparison_output": "comparison_output_path",
        "comparison_metadata_output": "comparison_metadata_output_path",
        "aligned_comparison_output": "aligned_comparison_output_path",
        "aligned_comparison_metadata_output": "aligned_comparison_metadata_output_path",
        "cluster_model_output": "cluster_model_output_path",
        "cluster_report_output": "cluster_report_output_path",
        "cluster_visualization_output": "cluster_visualization_output_path",
    }
    return {
        config_field: Path(raw_path)
        for output_arg, config_field in output_arg_to_config_field.items()
        if (raw_path := getattr(args, output_arg)) is not None
    }


def _config_overrides_from_args(args: argparse.Namespace) -> dict:
    optional_scalar_overrides = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "top_feature_count": args.top_feature_count,
        "training_embedding_pca_components": args.training_embedding_pca_components,
        "optuna_trials": args.optuna_trials,
        "train_ratio": args.train_ratio,
        "random_seed": args.random_seed,
        "aligned_comparison_start_date": args.aligned_start_date,
        "regression_style_fixed_horizon": args.regression_style_fixed_horizon,
    }
    overrides = {
        key: value
        for key, value in optional_scalar_overrides.items()
        if value is not None
    }

    if args.macro_tickers is not None:
        overrides["macro_tickers"] = _parse_macro_tickers(args.macro_tickers)
    if args.horizons is not None:
        overrides["horizon_candidates"] = _parse_horizon_candidates(args.horizons)
    if args.market_feature_columns is not None:
        overrides["market_feature_columns"] = _parse_market_feature_columns(
            args.market_feature_columns
        )
    if args.supplementary_ticker_feature_suffixes is not None:
        overrides["supplementary_ticker_feature_suffixes"] = (
            _parse_supplementary_ticker_feature_suffixes(
                args.supplementary_ticker_feature_suffixes
            )
        )

    overrides["market_news_only"] = args.market_news_only
    overrides.update(_optional_path_overrides(args))
    return overrides


def parse_args() -> argparse.Namespace:
    """
    shared 실행 엔트리포인트에서 사용할 CLI 인자를 정의한다.

    기본값만으로도 돌아가게 해두되, 팀원이 필요할 때는
    경로와 학습 범위를 쉽게 바꿀 수 있게 만드는 것이 목표다.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare two XGBoost target-ticker regression experiments: "
            "market-only vs market-plus-crawler-news."
        )
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument(
        "--ticker-preset",
        default="auto",
        help=(
            "Ticker preset to apply before CLI overrides. Use auto, none, or one of: "
            f"{', '.join(available_ticker_presets())}."
        ),
    )
    parser.add_argument(
        "--macro-tickers",
        default=None,
        help=(
            "Comma-separated macro tickers. If omitted, the ticker preset supplies them."
        ),
    )
    parser.add_argument(
        "--market-feature-columns",
        default=None,
        help=(
            "Comma-separated explicit market feature columns. If omitted, "
            "the ticker preset supplies the feature set."
        ),
    )
    parser.add_argument(
        "--supplementary-ticker-feature-suffixes",
        default=None,
        help=(
            "Comma-separated suffixes auto-added for non-fixed macro tickers, "
            "for example ret_5,ret_20,shock_5. Use none to rely only on explicit "
            "market feature columns."
        ),
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--news-input",
        default=None,
        help=(
            "News embedding CSV path. If omitted, the ticker-specific "
            "data/crawler/features/{ticker}/merged_finbert_with_embeddings.csv "
            "is used when available."
        ),
    )
    parser.add_argument(
        "--market-only-training-output",
        default=None,
    )
    parser.add_argument(
        "--market-only-predictions-output",
        default=None,
    )
    parser.add_argument("--market-only-model-output", default=None)
    parser.add_argument("--market-only-metadata-output", default=None)
    parser.add_argument(
        "--daily-news-output",
        default=None,
    )
    parser.add_argument(
        "--merged-training-output",
        default=None,
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
    )
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--comparison-output", default=None)
    parser.add_argument("--comparison-metadata-output", default=None)
    parser.add_argument(
        "--aligned-comparison-output",
        default=None,
    )
    parser.add_argument(
        "--aligned-comparison-metadata-output",
        default=None,
    )
    parser.add_argument("--cluster-model-output", default=None)
    parser.add_argument("--cluster-report-output", default=None)
    parser.add_argument("--cluster-visualization-output", default=None)
    parser.add_argument(
        "--horizons",
        default=None,
        help=(
            "Comma-separated horizon candidates, for example: 5,10,15,20. "
            "If omitted, the ticker preset supplies them."
        ),
    )
    parser.add_argument(
        "--aligned-start-date",
        default=None,
        help=(
            "Optional override for the fair-comparison start date. "
            "If omitted, the first trading day with lagged news coverage is used."
        ),
    )
    parser.add_argument("--top-feature-count", type=int, default=None)
    parser.add_argument(
        "--training-embedding-pca-components",
        type=int,
        default=None,
        help="Number of body_emb_* PCA components added to the main market_news training.",
    )
    parser.add_argument("--optuna-trials", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--regression-style-fixed-horizon",
        type=int,
        default=None,
        help=(
            "Fixed horizon used for the main shared experiments so they stay close to "
            "training/train_regression.py."
        ),
    )
    parser.add_argument(
        "--market-news-only",
        action="store_true",
        default=False,
        help="Skip market-only baseline and aligned comparison - run only market+news training.",
    )
    return parser.parse_args()


def _print_high_conf_report(metrics: dict) -> None:
    threshold = metrics.get("high_conf_threshold")
    long_acc = metrics.get("high_conf_long_accuracy")
    long_n = metrics.get("high_conf_long_count", 0)
    short_acc = metrics.get("high_conf_short_accuracy")
    short_n = metrics.get("high_conf_short_count", 0)
    if threshold is None:
        return
    print(f"  High confidence threshold: |Pred_LogRet| >= {threshold:.4f}")
    if long_acc is not None:
        print(f"  High-confidence long hit : {long_acc * 100:6.2f}%  (n={long_n})")
    if short_acc is not None:
        print(f"  High-confidence short hit: {short_acc * 100:6.2f}%  (n={short_n})")


def _section(title: str) -> None:
    print()
    print("=" * REPORT_WIDTH)
    print(title)
    print("-" * REPORT_WIDTH)


def _subsection(title: str) -> None:
    print()
    print(f"[{title}]")


def _format_rate(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:+.2f}%"


def _format_pct(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.2f}%"


def _print_key_values(rows: list[tuple[str, object]]) -> None:
    if not rows:
        return
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


def _print_signal_return_report(metrics: dict) -> None:
    if metrics.get("model_signal_avg_return_pct") is None:
        return

    _print_key_values(
        [
            (
                "Signal avg / cumulative",
                f"{_format_pct(metrics.get('model_signal_avg_return_pct'))} / "
                f"{_format_pct(metrics.get('model_signal_cumulative_return_pct'))}",
            ),
            (
                "Buy&hold avg / cumulative",
                f"{_format_pct(metrics.get('buy_and_hold_avg_return_pct'))} / "
                f"{_format_pct(metrics.get('buy_and_hold_cumulative_return_pct'))}",
            ),
            (
                "Long avg / Short avg",
                f"{_format_pct(metrics.get('model_signal_long_avg_return_pct'))} / "
                f"{_format_pct(metrics.get('model_signal_short_avg_return_pct'))}",
            ),
            (
                "High-conf signal avg",
                f"{_format_pct(metrics.get('model_signal_high_conf_avg_return_pct'))} "
                f"(n={metrics.get('model_signal_high_conf_count', 0)})",
            ),
        ]
    )


def _print_strategy_review_notes(config, predicted_groups: list[dict]) -> None:
    """
    Print the practical interpretation we use when reading this model's output.

    The numeric report above is deliberately model-centric. This block translates it
    into the trading overlay we discussed: T+5 forecast, next-day execution, and
    20 percentage-point incremental rebalancing.
    """
    ticker = config.target_ticker.upper()
    group_by_label = {group.get("label"): group for group in predicted_groups}
    fall_strong = group_by_label.get("fall_strong", {})
    rise_strong = group_by_label.get("rise_strong", {})

    _section("Strategy Notes")
    _print_key_values(
        [
            ("Forecast", "T+5 cumulative return"),
            ("Execution", "Use today's signal from the next trading day"),
            ("Rebalance", "buy threshold => +20%p, sell threshold => -20%p, otherwise hold"),
            ("Baseline", "Compare each ETF against its own Buy & Hold"),
            ("Threshold caution", "Full-period optimization is diagnostic; prefer walk-forward validation"),
        ]
    )

    if fall_strong:
        avg_actual = fall_strong.get("average_actual_return_pct")
        direction_accuracy = fall_strong.get("direction_accuracy")
        if avg_actual is not None:
            direction_str = (
                f"{direction_accuracy * 100:.1f}%"
                if direction_accuracy is not None
                else "N/A"
            )
            print(
                "  Strong down check : "
                f"n={fall_strong.get('count', 0)}, "
                f"avg actual={avg_actual:+.2f}%, hit={direction_str}"
            )
    if rise_strong:
        avg_actual = rise_strong.get("average_actual_return_pct")
        direction_accuracy = rise_strong.get("direction_accuracy")
        if avg_actual is not None:
            direction_str = (
                f"{direction_accuracy * 100:.1f}%"
                if direction_accuracy is not None
                else "N/A"
            )
            print(
                "  Strong up check   : "
                f"n={rise_strong.get('count', 0)}, "
                f"avg actual={avg_actual:+.2f}%, hit={direction_str}"
            )

    ticker_notes = {
        "QQQ": (
            "Upside signals are the useful part; be cautious with sell signals."
        ),
        "XLE": (
            "Prefer selective long entries; avoid mechanical selling from negative forecasts."
        ),
        "XLF": (
            "Regression ranking has been weak; keep Buy & Hold as the main baseline."
        ),
    }
    print(
        "  Ticker review     : "
        f"{ticker_notes.get(ticker, 'Validate thresholds ticker-by-ticker.')}"
    )


def _format_threshold(value: float) -> str:
    return "OFF" if abs(value) > 1.0 else f"{value * 100:+.2f}%"


def _print_high_volatility_zone_report(predicted_groups: list[dict]) -> None:
    group_by_label = {group.get("label"): group for group in predicted_groups}
    selected_groups = [
        group_by_label[label]
        for label in ("fall_strong", "rise_strong")
        if label in group_by_label
    ]
    if not selected_groups:
        return

    _section("High-Volatility Forecast Zones")
    print(f"  {'Zone':13s} {'N':>5s} {'Avg Pred':>10s} {'Avg Actual':>11s} {'Hit':>8s}")
    print("  " + "-" * 53)
    total_count = 0
    weighted_actual_sum = 0.0
    weighted_pred_sum = 0.0
    hit_sum = 0.0
    hit_count = 0

    for group in selected_groups:
        count = int(group.get("count", 0))
        avg_pred = group.get("average_predicted_return_pct")
        avg_actual = group.get("average_actual_return_pct")
        hit = group.get("direction_accuracy")
        print(
            f"  {group.get('label', 'N/A'):13s} {count:5d} "
            f"{_format_pct(avg_pred):>10s} {_format_pct(avg_actual):>11s} "
            f"{('N/A' if hit is None else f'{hit * 100:.1f}%'):>8s}"
        )
        total_count += count
        if avg_actual is not None:
            weighted_actual_sum += float(avg_actual) * count
        if avg_pred is not None:
            weighted_pred_sum += float(avg_pred) * count
        if hit is not None:
            hit_sum += float(hit) * count
            hit_count += count

    if total_count:
        combined_pred = weighted_pred_sum / total_count
        combined_actual = weighted_actual_sum / total_count
        combined_hit = None if hit_count == 0 else hit_sum / hit_count
        print(
            f"  {'strong_total':13s} {total_count:5d} "
            f"{_format_pct(combined_pred):>10s} {_format_pct(combined_actual):>11s} "
            f"{('N/A' if combined_hit is None else f'{combined_hit * 100:.1f}%'):>8s}"
        )


def _performance_from_equity(equity: pd.Series, daily_returns: pd.Series) -> dict:
    ending_value = float(equity.iloc[-1])
    active_returns = daily_returns.iloc[1:]
    drawdown = equity / equity.cummax() - 1.0
    volatility = (
        float(active_returns.std(ddof=1) * np.sqrt(252))
        if len(active_returns) > 1
        else 0.0
    )
    return {
        "return": ending_value - 1.0,
        "ending_value": ending_value,
        "max_drawdown": float(drawdown.min()),
        "volatility": volatility,
    }


def _run_incremental_rebalance_arrays(
    predicted_returns: np.ndarray,
    next_returns: np.ndarray,
    buy_hold_equity: pd.Series,
    buy_hold_daily_returns: pd.Series,
    sell_threshold: float,
    buy_threshold: float,
    step: float = 0.20,
) -> dict:
    weight = 0.0
    equity = 1.0
    equity_values = [1.0]
    daily_returns = [0.0]
    weights = []
    buy_count = 0
    sell_count = 0
    hold_count = 0

    for predicted_return, next_return in zip(predicted_returns, next_returns):
        if predicted_return >= buy_threshold:
            weight = min(1.0, weight + step)
            buy_count += 1
        elif predicted_return <= sell_threshold:
            weight = max(0.0, weight - step)
            sell_count += 1
        else:
            hold_count += 1

        strategy_return = weight * float(next_return)
        equity *= 1.0 + strategy_return
        equity_values.append(equity)
        daily_returns.append(strategy_return)
        weights.append(weight)

    strategy_equity = pd.Series(equity_values)
    strategy_daily_returns = pd.Series(daily_returns)

    strategy = _performance_from_equity(strategy_equity, strategy_daily_returns)
    buy_hold = _performance_from_equity(buy_hold_equity, buy_hold_daily_returns)
    return {
        "sell_threshold": sell_threshold,
        "buy_threshold": buy_threshold,
        "strategy": strategy,
        "buy_hold": buy_hold,
        "excess_return": strategy["return"] - buy_hold["return"],
        "average_weight": float(np.mean(weights)) if weights else 0.0,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
    }


def _prepare_rebalance_inputs(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    frame = predictions.copy()
    frame["Current_Date"] = pd.to_datetime(frame["Current_Date"], errors="coerce")
    frame = frame.dropna(subset=["Current_Date", "Current_Price"]).sort_values(
        "Current_Date"
    )
    frame = frame.reset_index(drop=True)
    frame["predicted_return"] = frame["Pred_Future_Price"] / frame["Current_Price"] - 1.0
    frame["next_return"] = frame["Current_Price"].pct_change(fill_method=None).shift(-1)

    return (
        frame["predicted_return"].to_numpy()[:-1],
        frame["next_return"].to_numpy()[:-1],
        frame["Current_Price"] / float(frame["Current_Price"].iloc[0]),
        frame["Current_Price"].pct_change(fill_method=None).fillna(0.0),
    )


def _find_best_incremental_rebalance_thresholds(predictions: pd.DataFrame) -> dict:
    (
        predicted_returns,
        next_returns,
        buy_hold_equity,
        buy_hold_daily_returns,
    ) = _prepare_rebalance_inputs(predictions)
    sell_values = np.round(np.arange(-0.20, 0.0001, 0.0005), 6)
    buy_values = np.round(np.arange(-0.01, 0.0201, 0.0005), 6)
    best_result: dict | None = None

    for sell_threshold in sell_values:
        for buy_threshold in buy_values:
            if sell_threshold >= buy_threshold:
                continue
            result = _run_incremental_rebalance_arrays(
                predicted_returns,
                next_returns,
                buy_hold_equity,
                buy_hold_daily_returns,
                float(sell_threshold),
                float(buy_threshold),
            )
            if best_result is None:
                best_result = result
                continue
            result_key = (
                result["strategy"]["return"],
                -(result["buy_count"] + result["sell_count"]),
            )
            best_key = (
                best_result["strategy"]["return"],
                -(best_result["buy_count"] + best_result["sell_count"]),
            )
            if result_key > best_key:
                best_result = result

    if best_result is None:
        raise ValueError("Could not find a valid threshold pair.")
    return best_result


def _print_threshold_strategy_report(predictions_path: Path) -> None:
    if not predictions_path.exists():
        return

    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
    if predictions.empty:
        return

    best = _find_best_incremental_rebalance_thresholds(predictions)
    _section("Threshold Rebalance Backtest")
    _print_key_values(
        [
            ("Search grid", "sell -20.00%..0.00%, buy -1.00%..+2.00%, step 0.05%p"),
            (
                "Best thresholds",
                f"sell={_format_threshold(best['sell_threshold'])}, "
                f"buy={_format_threshold(best['buy_threshold'])}",
            ),
            ("Execution", "T+5 forecast signal, next-trading-day rebalance, 20%p step"),
        ]
    )
    print()
    print(f"  {'Strategy':18s} {'Return':>10s} {'MDD':>10s} {'Avg W':>8s} {'B/S/H':>14s}")
    print("  " + "-" * 66)
    print(
        f"  {'Buy & Hold':18s} "
        f"{_format_rate(best['buy_hold']['return']):>10s} "
        f"{_format_rate(best['buy_hold']['max_drawdown']):>10s} "
        f"{'100.0%':>8s} {'-':>14s}"
    )
    trade_counts = f"{best['buy_count']}/{best['sell_count']}/{best['hold_count']}"
    print(
        f"  {'Model threshold':18s} "
        f"{_format_rate(best['strategy']['return']):>10s} "
        f"{_format_rate(best['strategy']['max_drawdown']):>10s} "
        f"{best['average_weight'] * 100:7.1f}% "
        f"{trade_counts:>14s}"
    )
    print(f"  {'Excess vs B&H':18s} {_format_rate(best['excess_return']):>10s}")
    print("  Note: this is full-period diagnostic optimization, not a validated live rule.")


def main() -> None:
    args = parse_args()

    config = make_training_config(
        ticker=args.target_ticker,
        news_input_path=(Path(args.news_input) if args.news_input is not None else None),
        preset=args.ticker_preset,
        **_config_overrides_from_args(args),
    )

    result = run_market_news_training_pipeline(config)
    market_news = result["market_news"]
    mean_baseline = result.get("mean_return_baseline", {})

    _section("Shared XGBoost Training Completed")
    _print_key_values(
        [
            ("Target ticker", config.target_ticker),
            ("Ticker preset", config.preset_name),
        ]
    )

    if mean_baseline:
        b_metrics = mean_baseline.get("metrics", {})
        _subsection("Mean Return Baseline")
        _print_key_values(
            [
                ("Direction accuracy", _format_rate(b_metrics.get("direction_accuracy"))),
                ("RMSE", f"{b_metrics.get('rmse', 0):.4f}"),
                ("Mean train logret", f"{b_metrics.get('mean_train_logret', 0):+.4f}"),
                ("Test rows", mean_baseline.get("test_rows", 0)),
            ]
        )

    if not config.market_news_only:
        market_only = result["market_only"]
        delta = result["delta_market_news_minus_market_only"]
        aligned = result["aligned_shared_period_comparison"]
        aligned_summary = aligned["summary"]

        _subsection("Market Only")
        _print_key_values(
            [
                ("Best horizon", f"{market_only['best_horizon']} trading days"),
                ("Selected features", market_only["selected_feature_count"]),
                ("RMSE", f"{market_only['metrics']['rmse']:.4f}"),
                (
                    "Direction accuracy",
                    _format_rate(market_only["metrics"].get("direction_accuracy")),
                ),
            ]
        )
        _print_high_conf_report(market_only["metrics"])
        _print_signal_return_report(market_only["metrics"])

    _subsection("Market + Crawler News")
    _print_key_values(
        [
            ("Best horizon", f"{market_news['best_horizon']} trading days"),
            ("Selected features", market_news["selected_feature_count"]),
            ("RMSE", f"{market_news['metrics']['rmse']:.4f}"),
            (
                "Direction accuracy",
                _format_rate(market_news["metrics"].get("direction_accuracy")),
            ),
        ]
    )
    _print_high_conf_report(market_news["metrics"])
    _print_signal_return_report(market_news["metrics"])

    if not config.market_news_only:
        _subsection("Delta: Market + News vs Market Only")
        delta_rows = [
            ("RMSE delta", f"{delta['rmse']:+.4f}"),
            ("Direction accuracy delta", _format_rate(delta["direction_accuracy"])),
        ]
        if "model_signal_avg_return_pct" in delta:
            delta_rows.append(
                ("Model signal avg return delta", _format_pct(delta["model_signal_avg_return_pct"]))
            )
        _print_key_values(delta_rows)

        _subsection("Fair Aligned Comparison")
        _print_key_values(
            [
                ("Aligned start date", aligned["aligned_start_date"]),
                (
                    "Best shared horizon",
                    f"{aligned_summary['best_shared_horizon_by_direction_accuracy_delta']} trading days",
                ),
                (
                    "Direction accuracy delta",
                    _format_rate(aligned_summary["direction_accuracy_delta"]),
                ),
                ("RMSE delta", f"{aligned_summary['rmse_delta']:+.4f}"),
            ]
        )

    cluster_report = (
        result.get("predicted_return_cluster_report")
        or result.get("volatility_prediction_report")
        or result.get("volatility_cluster_report", {})
    )
    predicted_groups = cluster_report.get("predicted_groups", [])
    if predicted_groups:
        metrics = cluster_report.get("source_model_metrics", {})
        _section(
            "Predicted Return Regimes "
            f"(T+{cluster_report.get('horizon', config.cluster_horizon)}, "
            f"{config.cluster_window_days}-calendar-day news window)"
        )
        if metrics:
            _print_key_values(
                [
                    ("Source direction accuracy", _format_rate(metrics.get("direction_accuracy"))),
                    ("Source RMSE", f"{metrics.get('rmse', 0.0):.4f}"),
                ]
            )
        print(
            f"  {'Label':13s} {'N':>5s} {'Avg Pred':>10s} {'Avg Actual':>11s} "
            f"{'Hit':>8s}  Date Range"
        )
        print("  " + "-" * 82)
        for group in predicted_groups:
            label = group["label"]
            count = group["count"]
            avg_pred_return = group.get("average_predicted_return_pct")
            avg_actual_return = group.get("average_actual_return_pct")
            avg_return_str = (
                "N/A"
                if avg_pred_return is None
                else f"{avg_pred_return:+.2f}%"
            )
            avg_actual_return_str = (
                "N/A"
                if avg_actual_return is None
                else f"{avg_actual_return:+.2f}%"
            )
            direction_accuracy = group.get("direction_accuracy")
            direction_str = (
                "N/A"
                if direction_accuracy is None
                else f"{direction_accuracy * 100:.1f}%"
            )
            dr = group.get("date_range", {})
            date_str = f"{dr.get('first', '?')} ~ {dr.get('last', '?')}" if dr else "N/A"
            print(
                f"  {label:13s} {count:5d} {avg_return_str:>10s} "
                f"{avg_actual_return_str:>11s} {direction_str:>8s}  {date_str}"
            )

    if predicted_groups:
        _print_high_volatility_zone_report(predicted_groups)

    _print_threshold_strategy_report(config.predictions_output_path)
    _print_strategy_review_notes(config, predicted_groups)

    _section("Saved Artifacts")
    rows = [
        ("Cluster model", config.cluster_model_output_path),
        ("Cluster report", config.cluster_report_output_path),
        ("Cluster visualization", config.cluster_visualization_output_path),
    ]
    if not config.market_news_only:
        rows = [
            ("Comparison CSV", config.comparison_output_path),
            ("Comparison JSON", config.comparison_metadata_output_path),
            ("Aligned CSV", config.aligned_comparison_output_path),
            ("Aligned JSON", config.aligned_comparison_metadata_output_path),
            *rows,
        ]
    _print_key_values(rows)


if __name__ == "__main__":
    main()

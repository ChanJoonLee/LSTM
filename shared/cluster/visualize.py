from __future__ import annotations

"""
뉴스 클러스터 시각화.

파이프라인에서 직접 호출하거나 단독으로 실행할 수 있다.

- 파이프라인 통합 (15일 창 실제 학습 벡터 사용):
      save_cluster_visualization(vectors, labels, counts, centroids, scaler, ...)

- 단독 실행 (daily_news_features 개별 행 사용):
      python shared/cluster/visualize.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 환경 — pyplot import 전에 선언해야 한다
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.cluster.model import (
    CLUSTER_BASE_FEATURE_COLS,
    CLUSTER_FEATURE_COLS,
    VOLATILITY_LABELS,
    build_event_dataset_with_embedding_pca,
    load_cluster_model,
)


_LABEL_COLORS: dict[str, str] = {
    "rise_strong": "#1a6e1a",
    "rise_mid":    "#2e7d32",
    "rise":        "#43a047",
    "neutral":     "#9e9e9e",
    "fall":        "#e53935",
    "fall_strong": "#7f0000",
}


def _assign_nearest_labels(
    points_scaled: np.ndarray,
    centroids_scaled: np.ndarray,
) -> list[str]:
    labels = []
    for pt in points_scaled:
        dists = np.linalg.norm(centroids_scaled - pt, axis=1)
        labels.append(VOLATILITY_LABELS[int(np.argmin(dists))])
    return labels


def _plot_scatter(
    ax: plt.Axes,
    pts_2d: np.ndarray,
    point_labels: list[str],
    cen_2d: np.ndarray,
    pca: PCA,
) -> None:
    for label in VOLATILITY_LABELS:
        mask = np.array([l == label for l in point_labels])
        if not mask.any():
            continue
        ax.scatter(
            pts_2d[mask, 0],
            pts_2d[mask, 1],
            c=_LABEL_COLORS[label],
            label=f"{label}  (n={int(mask.sum())})",
            alpha=0.4,
            s=18,
            linewidths=0,
        )

    for i, label in enumerate(VOLATILITY_LABELS):
        ax.scatter(
            cen_2d[i, 0],
            cen_2d[i, 1],
            c=_LABEL_COLORS[label],
            marker="*",
            s=420,
            edgecolors="black",
            linewidths=0.7,
            zorder=10,
        )
        ax.annotate(
            label,
            xy=(cen_2d[i, 0], cen_2d[i, 1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=_LABEL_COLORS[label],
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1  ({var1:.1f}% variance explained)", fontsize=10)
    ax.set_ylabel(f"PC2  ({var2:.1f}% variance explained)", fontsize=10)
    ax.set_title("All News Vectors + Cluster Centroids (★)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.75, borderpad=0.6)
    ax.grid(True, alpha=0.22)


def _plot_heatmap(
    ax: plt.Axes,
    centroids_scaled: np.ndarray,
    counts: np.ndarray,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> None:
    centroid_orig = scaler.inverse_transform(centroids_scaled)
    col_min = centroid_orig.min(axis=0)
    col_max = centroid_orig.max(axis=0)
    centroid_norm = (centroid_orig - col_min) / (col_max - col_min + 1e-9)

    im = ax.imshow(centroid_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=48, ha="right", fontsize=7)

    y_labels = [f"{lbl}  (n={cnt})" for lbl, cnt in zip(VOLATILITY_LABELS, counts)]
    ax.set_yticks(range(len(VOLATILITY_LABELS)))
    ax.set_yticklabels(y_labels, fontsize=8.5)

    for r in range(len(VOLATILITY_LABELS)):
        for c in range(len(feature_columns)):
            ax.text(
                c, r,
                f"{centroid_orig[r, c]:.2f}",
                ha="center", va="center",
                fontsize=5.5,
                color="black",
            )

    plt.colorbar(im, ax=ax, fraction=0.034, label="min-max normalized (per feature)")
    ax.set_title("Centroid Feature Profile", fontsize=10.5)


def save_cluster_visualization(
    vectors: np.ndarray,
    labels: list[str],
    counts: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler,
    output_path: Path,
    horizon: int = 15,
    window_days: int = 15,
    feature_columns: list[str] | None = None,
) -> None:
    """학습에 사용한 뉴스 창 벡터와 5개 중심점을 PCA 2D로 투영해 PNG로 저장한다."""
    vectors_scaled = scaler.transform(vectors)
    resolved_feature_columns = CLUSTER_FEATURE_COLS if feature_columns is None else feature_columns

    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([vectors_scaled, centroids]))

    pts_2d = pca.transform(vectors_scaled)
    cen_2d = pca.transform(centroids)

    fig = plt.figure(figsize=(19, 8))
    fig.suptitle(
        f"News Volatility Clusters — PCA 2D Projection\n"
        f"horizon={horizon}d · window={window_days}d · total vectors={len(vectors)}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.38)
    _plot_scatter(fig.add_subplot(gs[0]), pts_2d, labels, cen_2d, pca)
    _plot_heatmap(fig.add_subplot(gs[1]), centroids, counts, scaler, resolved_feature_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cluster visualization saved: {output_path}")


def main() -> None:
    """단독 실행 모드 — cluster_model.json 과 daily_news_features.csv 로 시각화를 재생성한다."""
    from shared.common.utils import training_data_path

    model_path = training_data_path("comparison", "qqq_volatility_cluster_model.json")
    news_path = training_data_path("market_news", "qqq_market_news_training_frame.csv")
    out_path = training_data_path("comparison", "qqq_cluster_visualization.png")

    with open(model_path, encoding="utf-8") as f:
        model_dict = json.load(f)

    centroids, scaler = load_cluster_model(model_dict)
    feature_columns = model_dict.get("feature_columns", CLUSTER_FEATURE_COLS)

    news_df = pd.read_csv(news_path, encoding="utf-8-sig")
    embedding_pca = model_dict.get("embedding_pca")
    if embedding_pca is not None:
        X_raw, point_labels, _dates, feature_columns, _embedding_pca = (
            build_event_dataset_with_embedding_pca(
                news_df,
                news_df,
                horizon=model_dict.get("horizon", 15),
                window_days=model_dict.get("window_days", 15),
                base_feature_columns=model_dict.get(
                    "base_feature_columns",
                    CLUSTER_BASE_FEATURE_COLS,
                ),
                embedding_pca=embedding_pca,
            )
        )
        counts = np.array(
            [sum(1 for l in point_labels if l == lbl) for lbl in VOLATILITY_LABELS],
            dtype=int,
        )
    else:
        X_raw = news_df[feature_columns].dropna().to_numpy(dtype=float)
        X_scaled = scaler.transform(X_raw)

        point_labels = _assign_nearest_labels(X_scaled, centroids)
        counts = np.array(
            [sum(1 for l in point_labels if l == lbl) for lbl in VOLATILITY_LABELS],
            dtype=int,
        )

    save_cluster_visualization(
        vectors=X_raw,
        labels=point_labels,
        counts=counts,
        centroids=centroids,
        scaler=scaler,
        output_path=out_path,
        horizon=model_dict.get("horizon", 15),
        window_days=model_dict.get("window_days", 15),
        feature_columns=feature_columns,
    )


if __name__ == "__main__":
    main()

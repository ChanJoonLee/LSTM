"""
통합 데이터 파이프라인: 정책 모니터링 데이터를 수집 후처리부터 감성분석, 임베딩까지 한 번에 처리.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path, feature_csv_path
from crawler.postprocessing.text_summarizer import summarize_to_under_limit as ollama_summarize
from crawler.postprocessing.sentiment_score import analyze_titles, analyze_bodies
from crawler.postprocessing.sentence_transformer import encode_summaries
from crawler.postprocessing.preprocessing import one_hot_encode_category

# 후처리 단계 설정
BODY_COL = "body"
BODY_SUMMARY_COL = "body_summary"
TITLE_COL = "title"
MAX_SUMMARY_CHARS = 10_000
SLEEP_BETWEEN_SUMMARIZE_SEC = 0.5
EMBEDDING_COL = f"{BODY_SUMMARY_COL}_embedding"
PCA_DIM = 30
EXPECTED_CATEGORY_VALUES = ["BIS", "EIA", "FOMC", "FRASER", "UCSB", "YAHOO"]


def _sector_pca_model_path(sector: str | None) -> str:
    """섹터별 PCA 모델 경로를 반환한다."""
    sector_key = str(sector or "").strip().lower()
    if not sector_key:
        return feature_csv_path("policy_updates_pca.pkl")
    return feature_csv_path(f"{sector_key}_merged_finbert_pca.pkl")


def _resolve_pca_components(embeddings: object, requested_components: int = PCA_DIM) -> int:
    """PCA에 사용할 실제 차원 수를 데이터 크기에 맞게 제한한다."""
    if not hasattr(embeddings, "shape"):
        return max(1, requested_components)

    n_samples, n_features = embeddings.shape
    return max(1, min(requested_components, n_samples, n_features))


def _load_or_fit_pca(embeddings, requested_components: int = PCA_DIM, pca_path: str | None = None):
    """기존 PCA 모델이 있으면 재사용하고, 없으면 새로 학습한다."""
    if pca_path is None:
        pca_path = _sector_pca_model_path(None)

    pca_file = Path(pca_path)
    if pca_file.exists():
        return joblib.load(pca_file), True

    n_components = _resolve_pca_components(embeddings, requested_components=requested_components)
    pca_model = PCA(n_components=n_components, random_state=42)
    pca_model.fit(embeddings)
    return pca_model, False


def _reduce_embeddings(embeddings, pca_model):
    """학습된 PCA 모델로 임베딩 차원을 축소한다."""
    return pca_model.transform(embeddings).astype("float32")


def apply_text_summarization(df: pd.DataFrame, body_col: str = BODY_COL, max_chars: int = MAX_SUMMARY_CHARS, sleep_sec: float = SLEEP_BETWEEN_SUMMARIZE_SEC) -> pd.DataFrame:
    """
    긴 본문을 요약한다.
    
    Args:
        df: 입력 데이터프레임
        body_col: 본문 컬럼명
        max_chars: 최대 문자 수 (이상이면 요약)
        sleep_sec: 요약 API 호출 간 대기 시간
    
    Returns:
        body_summary 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    df[body_col] = df[body_col].fillna("").astype(str)
    
    lengths = df[body_col].str.len()
    df["body_original_length"] = lengths
    df[BODY_SUMMARY_COL] = df[body_col]
    
    need_summary_mask = lengths >= max_chars
    indices = df.index[need_summary_mask].tolist()
    
    if indices:
        print(f"[UNIFIED] Text Summarization: {len(indices)} rows need summarization (>= {max_chars} chars)")
        
        for i, idx in enumerate(indices, start=1):
            text = df.at[idx, body_col]
            try:
                summary = ollama_summarize(text, limit_chars=max_chars)
                df.at[idx, BODY_SUMMARY_COL] = summary
            except Exception as e:
                print(f"[UNIFIED] WARN: summarize failed row={idx}: {e} -> truncating")
                df.at[idx, BODY_SUMMARY_COL] = text[:max_chars].rstrip()
            
            if i < len(indices):
                time.sleep(sleep_sec)
        
        print(f"[UNIFIED] Summarization complete: {len(indices)} rows summarized")
    else:
        print(f"[UNIFIED] Text Summarization: no rows need summarization")
    
    return df


def apply_one_hot_encoding(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    """
    카테고리 컬럼을 one-hot 인코딩한다.
    """
    df = df.copy()
    try:
        df = one_hot_encode_category(
            df,
            keep_category=True,
            prefix=f"{category_col}_",
            expected_categories=EXPECTED_CATEGORY_VALUES,
        )
        print(f"[UNIFIED] One-hot Encoding: applied to {category_col}")
    except Exception as e:
        print(f"[UNIFIED] WARN: one-hot encoding failed: {e}")
    
    return df


def apply_sentiment_analysis(df: pd.DataFrame, title_col: str = TITLE_COL, body_summary_col: str = BODY_SUMMARY_COL, batch_size: int = 8) -> pd.DataFrame:
    """
    제목과 본문에 감성 분석을 적용한다.
    """
    df = df.copy()
    
    if title_col not in df.columns:
        print(f"[UNIFIED] WARN: title column '{title_col}' not found, skipping sentiment analysis")
        return df
    
    if body_summary_col not in df.columns:
        print(f"[UNIFIED] WARN: body_summary column '{body_summary_col}' not found, skipping sentiment analysis")
        return df
    
    df[title_col] = df[title_col].fillna("").astype(str)
    df[body_summary_col] = df[body_summary_col].fillna("").astype(str)
    
    print(f"[UNIFIED] Sentiment Analysis: analyzing {len(df)} rows...")
    
    # 제목 분석
    title_results = analyze_titles(df[title_col].tolist(), batch_size=batch_size)
    for col, values in zip(["title_positive_prob", "title_negative_prob", "title_neutral_prob", "title_sentiment_score"], zip(*[r.values() for r in title_results])):
        df[col] = list(values)
    
    # 본문 분석
    body_results = analyze_bodies(df[body_summary_col].tolist(), max_chars=800, batch_size=batch_size)
    for col in ["body_positive_prob", "body_negative_prob", "body_neutral_prob", "body_sentiment_score", "body_n_chunks"]:
        df[col] = [r[col] for r in body_results]
    
    print(f"[UNIFIED] Sentiment Analysis: complete")
    
    return df


def apply_embeddings(df: pd.DataFrame, body_summary_col: str = BODY_SUMMARY_COL) -> pd.DataFrame:
    """
    본문 요약에 대해 임베딩을 생성한다.
    """
    df = df.copy()
    
    if body_summary_col not in df.columns:
        print(f"[UNIFIED] WARN: body_summary column '{body_summary_col}' not found, skipping embeddings")
        return df
    
    print(f"[UNIFIED] Embeddings: encoding {len(df)} rows...")

    if "sector" not in df.columns:
        df["sector"] = ""

    df["sector"] = df["sector"].fillna("").astype(str)
    df[EMBEDDING_COL] = None

    for sector_value, sector_df in df.groupby("sector", dropna=False, sort=False):
        sector_label = str(sector_value or "").strip()
        pca_path = _sector_pca_model_path(sector_label)

        summaries = sector_df[body_summary_col].fillna("").astype(str).tolist()
        embeddings = encode_summaries(summaries)

        pca_model, is_loaded = _load_or_fit_pca(embeddings, requested_components=PCA_DIM, pca_path=pca_path)
        reduced_embeddings = _reduce_embeddings(embeddings, pca_model)

        if not is_loaded:
            joblib.dump(pca_model, pca_path)
            print(f"[UNIFIED] PCA: saved new model to {pca_path}")

        reduced_vectors = reduced_embeddings.tolist()
        for row_index, vector in zip(sector_df.index, reduced_vectors):
            df.at[row_index, EMBEDDING_COL] = vector
        print(
            f"[UNIFIED] Embeddings: sector={sector_label or 'default'} "
            f"shape={embeddings.shape} -> PCA shape={reduced_embeddings.shape}"
        )
    
    return df


def apply_unified_pipeline(
    df: pd.DataFrame,
    include_summarization: bool = True,
    include_encoding: bool = True,
    include_sentiment: bool = True,
    include_embeddings: bool = True,
) -> pd.DataFrame:
    """데이터프레임에 통합 후처리 파이프라인을 적용합니다.

    인자:
        df: 처리할 pandas DataFrame.
        include_*: 각 처리 단계(include_summarization, include_encoding,
                   include_sentiment, include_embeddings)를 활성화하는 플래그.

    반환:
        처리된 pandas DataFrame. 이 함수는 파일을 저장하지 않으며,
        결과 저장은 호출자가 담당합니다.
    """
    if df is None:
        print("[UNIFIED] ERROR: input dataframe is None")
        return pd.DataFrame()

    df = df.copy()
    print(f"[UNIFIED] Loaded dataframe: {len(df)} rows, {len(df.columns)} columns")
    
    # 단계별 처리
    if include_summarization:
        print("[UNIFIED] Step 1/4: Text Summarization")
        df = apply_text_summarization(df)
    
    if include_encoding:
        print("[UNIFIED] Step 2/4: One-hot Encoding")
        df = apply_one_hot_encoding(df)
    
    if include_sentiment:
        print("[UNIFIED] Step 3/4: Sentiment Analysis")
        df = apply_sentiment_analysis(df)
    
    if include_embeddings:
        print("[UNIFIED] Step 4/4: Embeddings")
        df = apply_embeddings(df)
    
    return df


def main() -> None:
    """CLI 진입점: 기본값으로 전체 파이프라인 실행."""
    print("[UNIFIED] Starting unified postprocessing pipeline...")
    
    try:
        input_path = collected_csv_path("policy_updates_monitor.csv")
        if not Path(input_path).exists():
            print(f"[UNIFIED] ERROR: input file not found: {input_path}")
            return

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        df = apply_unified_pipeline(df=df)
        print(f"[UNIFIED] SUCCESS: processed {len(df)} rows")
    except Exception as e:
        print(f"[UNIFIED] FAILED: {e}")
        raise


if __name__ == "__main__":
    main()

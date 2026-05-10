from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

SAMPLE_SIZE = 5
RANDOM_SEED = 42


def parse_embedding_vector(embedding_val) -> np.ndarray | None:
    """
    CSV에 저장된 임베딩 벡터(리스트 형태)를 numpy 배열로 변환한다.
    
    Args:
        embedding_val: CSV에서 읽은 임베딩 값 (리스트 문자열 또는 리스트)
    
    Returns:
        numpy 배열 또는 None
    """
    if pd.isna(embedding_val):
        return None
    
    # 문자열 형태의 리스트를 eval로 파싱
    if isinstance(embedding_val, str):
        try:
            embedding_list = eval(embedding_val)
        except Exception:
            return None
    else:
        embedding_list = embedding_val
    
    if not isinstance(embedding_list, (list, tuple)):
        return None
    
    return np.array(embedding_list, dtype=np.float32)


def show_embedding_stats(embeddings: list[np.ndarray]) -> None:
    """
    임베딩 벡터들의 통계 정보를 출력한다.
    
    Args:
        embeddings: numpy 배열 리스트
    """
    valid_embeddings = [e for e in embeddings if e is not None]
    
    if not valid_embeddings:
        print("[WARNING] 유효한 임베딩을 찾을 수 없습니다.")
        return
    
    embeddings_array = np.array(valid_embeddings)
    
    print(f"\n[임베딩 벡터 통계]")
    print(f"유효한 벡터 개수: {len(valid_embeddings)}")
    print(f"벡터 차원: {embeddings_array.shape[1]}")
    print(f"벡터 범위: [{embeddings_array.min():.4f}, {embeddings_array.max():.4f}]")
    print(f"평균: {embeddings_array.mean():.6f}")
    print(f"표준편차: {embeddings_array.std():.6f}")
    print(f"norm 범위: [{np.linalg.norm(embeddings_array[0]):.4f}, {np.linalg.norm(embeddings_array[-1]):.4f}]")
    print("-" * 80)


def main() -> None:
    # ===== 요약문 샘플 =====
    csv_path = Path(PROJECT_ROOT) / "data" / "crawler" / "features" / "merged_finbert.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # body_summary 컬럼 확인
    if "body_summary" not in df.columns:
        raise ValueError(
            "body_summary 컬럼을 찾을 수 없습니다. "
            f"현재 컬럼: {', '.join(df.columns.astype(str))}"
        )
    
    # 비어있지 않은 요약문 필터링
    df_with_summary = df[
        df["body_summary"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    
    if df_with_summary.empty:
        raise ValueError("요약문이 있는 행을 찾을 수 없습니다.")
    
    # 랜덤 샘플 추출
    sample_size = min(SAMPLE_SIZE, len(df_with_summary))
    sample_df = df_with_summary.sample(n=sample_size, random_state=RANDOM_SEED)
    
    print(f"[MERGED FINBERT 요약문 샘플]")
    print(f"전체 행 수: {len(df)}, 요약문 있는 행: {len(df_with_summary)}")
    print(f"샘플 크기: {sample_size}\n")
    
    for index, (_, row) in enumerate(sample_df.iterrows(), 1):
        date = str(row.get("date", "(날짜 없음)"))
        category = str(row.get("category", "(카테고리 없음)"))
        title = str(row.get("title", "(제목 없음)"))
        summary = str(row.get("body_summary", "")).strip()
        
        print(f"\n[샘플 {index}]")
        print(f"날짜: {date}")
        print(f"카테고리: {category}")
        print(f"제목: {title}")
        print(f"길이: {len(summary)}자")
        print(f"요약문:\n{summary}\n")
        print("-" * 80)
    
    # ===== 임베딩 벡터 확인 =====
    embedded_csv_path = Path(PROJECT_ROOT) / "data" / "crawler" / "features" / "merged_finbert_with_embeddings.csv"
    
    if not embedded_csv_path.exists():
        print(f"\n[경고] 임베딩 파일을 찾을 수 없습니다: {embedded_csv_path}")
        return
    
    print(f"\n\n{'='*80}")
    print(f"[임베딩 벡터 확인]")
    print(f"{'='*80}")
    
    df_embedded = pd.read_csv(embedded_csv_path)
    
    if "body_summary_embedding" not in df_embedded.columns:
        print(f"[경고] body_summary_embedding 컬럼을 찾을 수 없습니다.")
        print(f"현재 컬럼: {', '.join(df_embedded.columns.astype(str))}")
        return
    
    # 임베딩 벡터 파싱
    embeddings_list = []
    for emb_val in df_embedded["body_summary_embedding"]:
        emb = parse_embedding_vector(emb_val)
        if emb is not None:
            embeddings_list.append(emb)
    
    if not embeddings_list:
        print("[경고] 유효한 임베딩을 찾을 수 없습니다.")
        return
    
    print(f"\n전체 임베딩 개수: {len(df_embedded)}")
    print(f"유효한 임베딩: {len(embeddings_list)}")
    
    # 임베딩 통계 출력
    show_embedding_stats(embeddings_list)
    
    # 샘플 임베딩 벡터 출력
    print(f"\n[샘플 임베딩 벡터 (처음 3개)]")
    for idx, emb in enumerate(embeddings_list[:3], 1):
        print(f"\n벡터 {idx}:")
        print(f"  형태: {emb.shape}")
        print(f"  첫 10개 값: {emb[:10]}")
        print(f"  norm: {np.linalg.norm(emb):.6f}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path, summarized_csv_path


DEFAULT_MERGED_OUTPUT_CSV = feature_csv_path("merged_table_sorted.csv")
DEFAULT_ENCODED_OUTPUT_CSV = feature_csv_path("merged_table_sorted_encoded.csv")
# Note: cyclical time features removed per request
DATE_COL = "date"
BODY_COL = "body"
BODY_LENGTH_COL = "body_original_length"


def _existing_csv_paths(csv_paths: Iterable[str]) -> list[str]:
    return [path for path in csv_paths if Path(path).exists()]


def _pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_date_series(s: pd.Series) -> pd.Series:
    """
    Normalize a date-like series to 'YYYY-MM-DD' strings when possible.
    If parsing fails, keep the original non-empty string value.
    """
    # Convert to string and handle missing values
    raw = s.fillna("").astype(str).str.strip()
    raw_clean = raw.where(raw != "", other=pd.NA)

    # Parse dates
    dt = pd.to_datetime(raw_clean, errors="coerce")
    # Format valid dates as YYYY-MM-DD, keep original for parsing failures
    out = dt.dt.strftime("%Y-%m-%d")
    out = out.where(~dt.isna(), other=raw_clean)
    return out


def merge_csvs_to_table(
    csv_paths: List[str],
    encoding: str = "utf-8-sig",
    drop_duplicates: bool = True,
    sort_by_date: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Read multiple CSV files and merge them into a standardized table.

    Output columns:
    - date: date / release_date / published_date
    - category: category
    - doc_type: doc_type
    - title: title
    - body: body
    - body_summary: summarized body text when available
    - link: link / url
    """
    tables: List[pd.DataFrame] = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path, encoding=encoding)
        except Exception as e:
            raise ValueError(f"Failed to read CSV {path}: {e}")

        date_col = _pick_first_existing(df, ["date", "release_date", "published_date"])
        category_col = _pick_first_existing(df, ["category"])
        doc_type_col = _pick_first_existing(df, ["doc_type"])
        title_col = "title" if "title" in df.columns else None
        body_col = _pick_first_existing(df, ["body"])
        summary_col = _pick_first_existing(df, ["body_summary", "summary"])
        link_col = _pick_first_existing(df, ["link", "url"])

        missing = [
            name
            for name, col in [
                ("date", date_col),
                ("category", category_col),
                ("doc_type", doc_type_col),
                ("title", title_col),
                ("body", body_col),
                ("link", link_col),
            ]
            if col is None
        ]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        body_series = df[body_col].fillna("").astype(str)
        if summary_col is not None:
            summary_series = df[summary_col].fillna("").astype(str)
        else:
            summary_series = pd.Series([""] * len(df), index=df.index, dtype="object")

        # Use existing body_original_length if available, otherwise calculate
        if BODY_LENGTH_COL in df.columns:
            body_length_series = pd.to_numeric(df[BODY_LENGTH_COL], errors="coerce").fillna(body_series.str.len())
        else:
            body_length_series = body_series.str.len()

        out = pd.DataFrame(
            {
                "date": _normalize_date_series(df[date_col]),
                "category": df[category_col],
                "doc_type": df[doc_type_col],
                "title": df[title_col],
                BODY_COL: body_series,
                "body_summary": summary_series,
                BODY_LENGTH_COL: body_length_series,
                "link": df[link_col],
            }
        )
        tables.append(out)

    merged = pd.concat(tables, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates()

    merged = merged[["date", "category", "doc_type", "title", BODY_COL, "body_summary", BODY_LENGTH_COL, "link"]]

    if sort_by_date:
        # Convert date column to datetime for proper sorting
        date_numeric = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.iloc[date_numeric.values.argsort()[::-1]].reset_index(drop=True)

    return merged


def one_hot_encode_category(
    df: pd.DataFrame,
    keep_category: bool = True,
    prefix: str = "category",
    dtype: str = "int64",
) -> pd.DataFrame:
    """
    One-hot encode the `category` column from an existing DataFrame.
    """
    if "category" not in df.columns:
        raise ValueError(
            "`category` column was not found. "
            f"Available columns: {list(df.columns)}"
        )

    encoded = pd.get_dummies(df["category"], prefix=prefix, dtype=dtype)

    if keep_category:
        return pd.concat([df, encoded], axis=1)

    return pd.concat([df.drop(columns=["category"]), encoded], axis=1)


def read_csv_and_one_hot_encode_category(
    csv_path: str,
    encoding: str = "utf-8-sig",
    keep_category: bool = True,
    prefix: str = "category",
    dtype: str = "int64",
) -> pd.DataFrame:
    """
    Read a CSV file and one-hot encode the `category` column.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    return one_hot_encode_category(
        df,
        keep_category=keep_category,
        prefix=prefix,
        dtype=dtype,
    )


def main() -> None:
    csv_candidates = [
        summarized_csv_path("fed_fomc_links_summarized.csv"),
        summarized_csv_path("ucsb_presidential_documents_summarized.csv"),
        summarized_csv_path("bis_press_releases_summarized.csv"),
    ]
    csv_paths = _existing_csv_paths(csv_candidates)

    if not csv_paths:
        raise FileNotFoundError(
            "No summarized crawler outputs were found. "
            f"Checked: {csv_candidates}"
        )

    merged = merge_csvs_to_table(csv_paths)
    print("[INFO] merged_rows=", len(merged))
    merged.to_csv(DEFAULT_MERGED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved merged file: {DEFAULT_MERGED_OUTPUT_CSV}")

    encoded = one_hot_encode_category(merged, keep_category=True)
    encoded.to_csv(DEFAULT_ENCODED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved encoded file: {DEFAULT_ENCODED_OUTPUT_CSV}")


if __name__ == "__main__":
    main()

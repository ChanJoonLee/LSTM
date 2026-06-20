from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.collectors.bis import create_requests_session, crawl_bis_news_index_selenium, extract_article
from crawler.collectors.fed import crawl_implementation_note, crawl_fomc_statement, crawl_minutes
from crawler.collectors.ucsb import (
    DOC_TYPE_URLS,
    DEFAULT_KEYWORD_CONFIG_PATH,
    crawl_listing,
    load_keyword_dictionary,
    parse_article,
)
from crawler.postprocessing.unified_pipeline import apply_unified_pipeline
from crawler.support_legacy.data_paths import collected_csv_path, feature_csv_path

BASE_URL = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"

DEFAULT_OUTPUT_CSV = collected_csv_path("policy_updates_monitor.csv")
DEFAULT_PROCESSED_OUTPUT_CSV = feature_csv_path("policy_updates_features.csv")
DEFAULT_INTERVAL_SEC = 86400
US_EASTERN_TZ = ZoneInfo("America/New_York")

CANONICAL_COLUMNS = [
    # 새 기록과 기존 기록을 같은 스키마로 합치기 위한 표준 컬럼 순서.
    "source",
    "category",
    "doc_type",
    "published_date",
    "release_date",
    "title",
    "url",
    "body",
    "matched_keyword_groups",
    "matched_keywords",
    "collected_at",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _parse_us_date(raw_value: str | None) -> date | None:
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%B %d, %Y").date()
    except ValueError:
        return None


def _format_iso_date(raw_value: str | None) -> str:
    parsed = _parse_us_date(raw_value)
    return parsed.isoformat() if parsed else ""


def _us_eastern_now() -> datetime:
    # 모든 날짜 기준을 미국 동부시간으로 통일한다.
    return datetime.now(US_EASTERN_TZ)


def _target_policy_news_date(reference_dt: datetime | None = None) -> date:
    current_dt = reference_dt or _us_eastern_now()
    # 모니터링 대상은 "현재 시점의 전날"로 고정한다.
    return current_dt.date() - timedelta(days=1)


def _seconds_until_next_us_eastern_midnight(reference_dt: datetime | None = None) -> float:
    current_dt = reference_dt or _us_eastern_now()
    # 다음 자정까지 남은 초를 계산해 하루 1회 실행 주기를 만든다.
    next_midnight = current_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(0.0, (next_midnight - current_dt).total_seconds())


def _existing_frame(output_csv: str) -> pd.DataFrame:
    path = Path(output_csv)
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalise_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="last")

    return df


def _collect_fomc_records(target_date: date) -> list[dict[str, Any]]:
    # FOMC는 캘린더 페이지를 읽고 statement, minutes, implementation note를 구분한다.
    response = requests.get(FOMC_CALENDAR_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    records: list[dict[str, Any]] = []

    sections = soup.find_all("div", class_="panel-default")
    # 최신 연도부터 오래된 연도 순으로 순회한다.
    for section in reversed(sections):
        heading = section.find("h4")
        if heading is None:
            continue

        heading_text = heading.get_text(" ", strip=True)
        if not re.match(r"(\d{4}) FOMC Meetings", heading_text):
            continue

        meetings = section.find_all("div", class_="fomc-meeting")

        for meeting in meetings:
            for link in meeting.find_all("a", href=True):
                label = _clean_text(link.get_text(" ", strip=True)).lower()
                url = urljoin(BASE_URL, link["href"])

                doc_type = None
                article: dict[str, Any] | None = None

                if "implementation note" in label:
                    doc_type = "implementation_note"
                    article = crawl_implementation_note(url)
                elif label == "html":
                    parent_strong = link.parent.find("strong") if link.parent else None
                    if parent_strong is None:
                        continue

                    parent_title = _clean_text(parent_strong.get_text(" ", strip=True)).lower()

                    if "statement:" in parent_title:
                        doc_type = "statement"
                        article = crawl_fomc_statement(url)
                    elif "minutes:" in parent_title:
                        doc_type = "minutes"
                        article = crawl_minutes(url)

                        release_match = re.search(
                            r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                            link.parent.get_text(" ", strip=True),
                        )
                        if release_match:
                            article["release_date"] = release_match.group(1)

                if not doc_type or not article:
                    continue

                published_date_value = _parse_us_date(article.get("release_date"))
                if published_date_value != target_date:
                    continue

                records.append(
                    {
                        "source": "FOMC",
                        "category": "FOMC",
                        "doc_type": doc_type,
                        "published_date": _format_iso_date(article.get("release_date")),
                        "release_date": article.get("release_date", ""),
                        "title": article.get("title", ""),
                        "url": url,
                        "body": article.get("body", ""),
                        "matched_keyword_groups": "",
                        "matched_keywords": "",
                        "collected_at": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )

    return records


def _collect_bis_records(target_date: date, max_pages: int, sleep_sec: float) -> list[dict[str, Any]]:
    # BIS는 목록 페이지에서 링크를 모은 뒤 상세 페이지를 다시 요청한다.
    listing_items = crawl_bis_news_index_selenium(max_pages=max_pages, sleep_sec=sleep_sec)
    if not listing_items:
        return []

    session = create_requests_session()
    records: list[dict[str, Any]] = []
    # listing_items are expected newest->oldest; no need for consecutive counters

    for item in listing_items:
        article = extract_article(item["url"], session=session, sleep_sec=sleep_sec)
        if not article:
            continue

        # 전날 기사만 CSV에 남기기 위해 날짜 기준으로 걸러낸다.
        published_date_value = None
        published_date_raw = str(article.get("published_date", "")).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", published_date_raw):
            try:
                published_date_value = datetime.strptime(published_date_raw, "%Y-%m-%d").date()
            except ValueError:
                published_date_value = None

        if published_date_value is None:
            continue

        # 목록은 최신순으로 내려오기 때문에, 타겟 날짜보다 이전인 기사를 만나면
        # 그 이후의 항목들도 모두 이전 날짜일 가능성이 높다. 즉시 중단해도 안전하다.
        if published_date_value < target_date:
            break

        # 타겟 날짜보다 미래(더 최근)인 것은 건너뛴다.
        if published_date_value > target_date:
            continue

        records.append(
            {
                "source": "BIS",
                "category": article.get("category", "BIS"),
                "doc_type": article.get("doc_type", "press_release"),
                "published_date": article.get("published_date", ""),
                "release_date": "",
                "title": article.get("title", ""),
                "url": item["url"],
                "body": article.get("body", ""),
                "matched_keyword_groups": "",
                "matched_keywords": "",
                "collected_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )

    return records


def _collect_ucsb_records(
    target_date: date,
    sleep_sec: float,
    keyword_config_path: str | Path | None,
    doc_types: list[str] | None,
) -> list[dict[str, Any]]:
    # UCSB는 키워드 매칭이 끝난 문서만 모니터링 CSV에 넣는다.
    keyword_dictionary = load_keyword_dictionary(keyword_config_path or DEFAULT_KEYWORD_CONFIG_PATH)
    selected_doc_types = list(doc_types or DOC_TYPE_URLS.keys())
    invalid_doc_types = [doc_type for doc_type in selected_doc_types if doc_type not in DOC_TYPE_URLS]
    if invalid_doc_types:
        raise ValueError(f"Unsupported doc types: {invalid_doc_types}")

    records: list[dict[str, Any]] = []

    for doc_type in selected_doc_types:
        listing_items = crawl_listing(
            base_url=DOC_TYPE_URLS[doc_type],
            doc_type=doc_type,
            start_date=target_date,
            sleep_sec=sleep_sec,
        )

        for item in listing_items:
            article = parse_article(item, keyword_dictionary=keyword_dictionary)
            if article is None:
                continue

            published_date_value = _parse_us_date(article.get("published_date"))
            if published_date_value != target_date:
                continue

            records.append(
                {
                    "source": "UCSB",
                    "category": article.get("category", "UCSB Presidency Project"),
                    "doc_type": article.get("doc_type", doc_type),
                    "published_date": article.get("published_date", ""),
                    "release_date": "",
                    "title": article.get("title", ""),
                    "url": item["url"],
                    "body": article.get("body", ""),
                    "matched_keyword_groups": article.get("matched_keyword_groups", ""),
                    "matched_keywords": article.get("matched_keywords", ""),
                    "collected_at": datetime.utcnow().isoformat(timespec="seconds"),
                }
            )

        time.sleep(sleep_sec)

    return records


def collect_policy_updates(
    bis_max_pages: int = 5,
    sleep_sec: float = 1.0,
    keyword_config_path: str | Path | None = None,
    doc_types: list[str] | None = None,
    target_date: date | None = None,
) -> pd.DataFrame:
    # 이전 파일을 읽지 않고, 이번 사이클에서 새로 수집된 레코드만 반환한다.
    target_date_value = target_date or _target_policy_news_date()

    new_records: list[dict[str, Any]] = []
    seen_urls = set()

    print(f"[MONITOR] Starting FOMC crawl for {target_date_value.isoformat()}")
    for record in _collect_fomc_records(target_date_value):
        if record["url"] in seen_urls:
            continue
        seen_urls.add(record["url"])
        new_records.append(record)

    print(f"[MONITOR] Starting BIS crawl for {target_date_value.isoformat()} (max_pages={bis_max_pages})")
    for record in _collect_bis_records(target_date_value, max_pages=bis_max_pages, sleep_sec=sleep_sec):
        if record["url"] in seen_urls:
            continue
        seen_urls.add(record["url"])
        new_records.append(record)

    print(f"[MONITOR] Starting UCSB crawl for {target_date_value.isoformat()}")
    for record in _collect_ucsb_records(
        target_date_value,
        sleep_sec=sleep_sec,
        keyword_config_path=keyword_config_path,
        doc_types=doc_types,
    ):
        if record["url"] in seen_urls:
            continue
        seen_urls.add(record["url"])
        new_records.append(record)

    new_df = _normalise_records(new_records)

    # 반환: 이번 사이클에서 새로 수집된 행들(중복 제거된 상태)
    return new_df


def run_postprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """주어진 DataFrame에 통합 후처리를 적용하고 처리된 DataFrame을 반환한다."""
    print("[MONITOR] Starting unified postprocessing pipeline...")
    try:
        result_df = apply_unified_pipeline(
            df=df,
            include_summarization=True,
            include_encoding=True,
            include_sentiment=True,
            include_embeddings=True,
        )
        print(f"[MONITOR] Postprocessing pipeline completed: {len(result_df)} rows processed")
        return result_df
    except Exception as e:
        print(f"[MONITOR] ERROR: postprocessing pipeline failed: {e}")
        raise


def run_monitor(
    output_csv: str = DEFAULT_PROCESSED_OUTPUT_CSV,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    bis_max_pages: int = 5,
    sleep_sec: float = 1.0,
    keyword_config_path: str | Path | None = None,
    doc_types: list[str] | None = None,
    max_cycles: int | None = None,
) -> None:
    cycle = 0

    while True:
        target_date_value = datetime(2026, 3, 18).date()
        # target_date_value = _target_policy_news_date()
        cycle += 1
        # 각 주기마다 전날 데이터를 한 번만 모은 뒤, 기본값이면 다음 자정까지 쉰다.
        new_df = collect_policy_updates(
            bis_max_pages=bis_max_pages,
            sleep_sec=sleep_sec,
            keyword_config_path=keyword_config_path,
            doc_types=doc_types,
            target_date=target_date_value,
        )

        new_row_count = len(new_df)
        if new_row_count > 0:
            print(f"[MONITOR] Applying unified pipeline to {new_row_count} new rows")
            processed_new = run_postprocessing_pipeline(df=new_df)

            processed_path = Path(output_csv)
            if processed_path.exists():
                try:
                    existing_processed = pd.read_csv(processed_path, encoding="utf-8-sig")
                except Exception:
                    existing_processed = pd.DataFrame()
            else:
                existing_processed = pd.DataFrame()

            if existing_processed.empty:
                combined_processed = processed_new.copy()
            else:
                combined_processed = pd.concat([existing_processed, processed_new], ignore_index=True, sort=False)

            if not combined_processed.empty and "url" in combined_processed.columns:
                combined_processed = combined_processed.drop_duplicates(subset=["url"], keep="last")

            combined_processed_path = Path(output_csv)
            combined_processed_path.parent.mkdir(parents=True, exist_ok=True)
            combined_processed.to_csv(combined_processed_path, index=False, encoding="utf-8-sig")

            print(f"[MONITOR] Saved processed output: {combined_processed_path} rows={len(combined_processed)}")
        else:
            print("[MONITOR] No new data collected, skipping processing and output save")

        print(
            f"[MONITOR] cycle={cycle} target_date={target_date_value.isoformat()} "
            f"new_rows={new_row_count} processed_output={output_csv}"
        )

        if max_cycles is not None and cycle >= max_cycles:
            break

        if interval_sec <= 0:
            break

        if interval_sec >= DEFAULT_INTERVAL_SEC:
            time.sleep(_seconds_until_next_us_eastern_midnight())
        else:
            time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor FOMC, BIS, and UCSB policy updates for the previous America/New_York day")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_PROCESSED_OUTPUT_CSV, help="Processed output CSV path (features)")
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help="Seconds between cycles; default schedules the next run at the next America/New_York midnight",
    )
    parser.add_argument("--bis-max-pages", type=int, default=1, help="Maximum BIS listing pages to scan per cycle")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Delay between source requests")
    parser.add_argument("--keyword-config-path", type=str, default=str(DEFAULT_KEYWORD_CONFIG_PATH), help="UCSB keyword config path")
    parser.add_argument(
        "--doc-types",
        nargs="*",
        default=None,
        help="Optional UCSB document types to monitor",
    )
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional upper bound for repeated monitoring cycles")
    args = parser.parse_args()

    run_monitor(
        output_csv=args.output_csv,
        interval_sec=args.interval_sec,
        bis_max_pages=args.bis_max_pages,
        sleep_sec=args.sleep_sec,
        keyword_config_path=args.keyword_config_path,
        doc_types=args.doc_types,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
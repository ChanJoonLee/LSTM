import time
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path

BASE_URL = "https://www.federalreserve.gov"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# 2017-2020 FOMC 캘린더 페이지 URL 맵핑
HISTORICAL_CALENDAR_URLS = {
    2017: f"{BASE_URL}/monetarypolicy/fomchistorical2017.htm",
    2018: f"{BASE_URL}/monetarypolicy/fomchistorical2018.htm",
    2019: f"{BASE_URL}/monetarypolicy/fomchistorical2019.htm",
    2020: f"{BASE_URL}/monetarypolicy/fomchistorical2020.htm",
}


def crawl_fomc_statement(url: str) -> dict:
    """
    FOMC statement 상세 페이지에서
    공개일, 제목, 본문 텍스트를 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 페이지 상단 게시 날짜
    date_tag = soup.find("p", class_="article__time")
    release_date = date_tag.get_text(" ", strip=True) if date_tag else ""

    # statement 제목
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # statement 본문은 보통 col-sm-8 폭의 본문 영역에 들어 있다.
    divs = soup.find_all("div", class_="col-sm-8")

    article = None
    for div in divs:
        classes = div.get("class", [])
        if "heading" not in classes:
            article = div
            break

    contents = []
    stop_texts = ("for media inquiries", "implementation note issued")

    if article:
        # 본문 전체를 한 번에 가져와 라인 단위로 스톱 구간 전까지만 수집한다.
        raw = article.get_text("\n", strip=True)
        for line in (ln.strip() for ln in raw.splitlines()):
            if not line:
                continue

            lowered = line.lower()
            if lowered.startswith(stop_texts):
                break

            contents.append(line)

    body_text = " ".join(contents)

    return {
        "release_date": release_date,
        "title": title,
        "body": body_text
    }


def crawl_minutes(url: str) -> dict:
    """
    FOMC minutes 상세 페이지에서
    제목과 본문 텍스트를 추출한다.

    release_date는 상세 페이지가 아니라 캘린더 페이지의
    '(Released ...)' 문구에서 따로 채운다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # minutes 전체 본문이 들어 있는 컨테이너
    article = soup.find("div", id="article")

    # 제목은 본문 컨테이너 안의 h3에서 읽는다.
    title_tag = article.find("h3") if article else None
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    contents = []

    if article:
        # 전체 텍스트를 라인 단위로 읽어 본문만 수집한다.
        raw = article.get_text("\n", strip=True)
        for line in (ln.strip() for ln in raw.splitlines()):
            if not line:
                continue

            contents.append(line)

    body_text = " ".join(contents)

    return {
        "release_date": None,
        "title": title,
        "body": body_text
    }


def main() -> None:
    results = []

    # 2017-2020년 각 연도별로 캘린더 페이지를 크롤링한다.
    for year, calendar_url in sorted(HISTORICAL_CALENDAR_URLS.items()):
        print(f"\n[{year}] 크롤링 시작: {calendar_url}")

        try:
            # FOMC 연간 캘린더 페이지를 가져온다.
            response = requests.get(calendar_url, headers=HEADERS, timeout=20)
            response.raise_for_status()

            # 캘린더 HTML을 파싱한다.
            soup = BeautifulSoup(response.text, "html.parser")

            # 연도별 패널을 순회하며 "2020 FOMC Meetings" 같은 섹션을 고른다.
            meetings = soup.find_all("div", class_="panel-padded")

            year_count = 0

            for meeting in meetings:
                # 회의 블록 안의 문서 링크를 순회한다.
                for link in meeting.find_all("a", href=True):
                    label = link.get_text(" ", strip=True).lower()
                    url = urljoin(BASE_URL, link["href"])

                    doc_type = None
                    article = None

                    # HTML 링크는 부모 strong 텍스트를 보고
                    # statement인지 minutes인지 구분한다.
                    if label == "statement":
                        doc_type = "statement"
                        try:
                            article = crawl_fomc_statement(url)
                        except Exception as e:
                            print(f"  [WARN] Statement 파싱 실패 ({url}): {e}")
                            continue

                    elif label == "html":
                        parent_title = link.parent.get_text(" ", strip=True) if link.parent else None
                        if not parent_title.startswith("Minutes"):
                            continue
                        doc_type = "minutes"
                        try:
                            article = crawl_minutes(url)

                            # minutes 공개일은 상세 페이지보다 캘린더 문구가 더 명확해서
                            # 현재 링크가 속한 블록의 Released 문구에서 추출한다.
                            release_match = re.search(
                                r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                                link.parent.get_text(" ", strip=True)
                            )

                            release_date = release_match.group(1) if release_match else None
                            article["release_date"] = release_date
                        except Exception as e:
                            print(f"  [WARN] Minutes 파싱 실패 ({url}): {e}")
                            continue

                    # 문서 파싱이 성공한 경우 표준 레코드로 저장한다.
                    if doc_type and article:
                        results.append({
                            "release_date": article["release_date"],
                            "category": "FOMC",
                            "doc_type": doc_type,
                            "url": url,
                            "title": article["title"],
                            "body": article["body"]
                        })
                        year_count += 1

                    # 연속 요청 부담을 줄이기 위해 짧게 쉰다.
                    time.sleep(0.5)

            print(f"[{year}] 크롤링 완료: {year_count}건 수집")

        except Exception as e:
            print(f"[{year}] 크롤링 실패: {e}")
            continue

    # 결과를 DataFrame으로 정리한다.
    new_df = pd.DataFrame(results)
    print(f"\n새로 수집된 데이터: {len(new_df)}건")

    # 기존 fed_fomc_links.csv가 있으면 읽어서 병합한다.
    existing_csv = collected_csv_path("fed_fomc_links.csv")
    existing_path = Path(existing_csv)

    if existing_path.exists():
        existing_df = pd.read_csv(existing_csv, encoding="utf-8-sig")
        print(f"기존 데이터: {len(existing_df)}건")
        
        # 새로운 데이터와 기존 데이터를 이어붙인다.
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"병합 후 (중복 제거 전): {len(merged_df)}건")
        
        # 중복 제거 (url 기준)
        merged_df = merged_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        print(f"중복 제거 후: {len(merged_df)}건")
    else:
        print("기존 데이터 없음. 새 데이터만 저장합니다.")
        merged_df = new_df.drop_duplicates().reset_index(drop=True)

    print(f"\n문서 유형 분포:\n{merged_df['doc_type'].value_counts()}")
    print(f"\n샘플:\n{merged_df.head(10)}")

    # 최신순으로 정렬 (release_date 기준, 최신이 위)
    merged_df['_sort_date'] = pd.to_datetime(merged_df['release_date'], errors='coerce')
    merged_df = merged_df.sort_values(by='_sort_date', ascending=False, na_position='last').reset_index(drop=True)
    merged_df = merged_df.drop(columns=['_sort_date'])

    # 최종 결과를 fed_fomc_links.csv에 저장한다.
    output_csv = collected_csv_path("fed_fomc_links.csv")
    merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_csv}")


if __name__ == "__main__":
    main()
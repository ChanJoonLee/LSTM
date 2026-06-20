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
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def crawl_implementation_note(url: str) -> dict:
    """
    Implementation Note 상세 페이지에서
    공개일, 제목, 본문 텍스트를 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 페이지 상단에 노출된 공식 게시 날짜를 읽는다.
    date_tag = soup.find("p", class_="article__time")
    release_date = date_tag.get_text(" ", strip=True) if date_tag else ""

    # 제목은 보통 h3에 들어 있다.
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # 같은 폭의 div가 여러 개 있을 수 있으므로,
    # heading 블록을 제외한 첫 번째 본문 컨테이너를 사용한다.
    divs = soup.find_all("div", class_="col-xs-12")

    article = None
    for div in divs:
        classes = div.get("class", [])
        if "heading" not in classes:
            article = div
            break

    contents = []

    if article:
        # 컨테이너 전체를 한 번에 텍스트로 뽑아 라인 단위로 처리한다.
        raw = article.get_text("\n", strip=True)
        for line in (ln.strip() for ln in raw.splitlines()):
            if not line:
                continue
            contents.append(line)

    body_text = " ".join(contents)

    return {
        "release_date": release_date,
        "title": title,
        "body": body_text
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
        # 전체 텍스트를 라인 단위로 읽어 숫자 줄 제거, 'notation vote' 이전까지만 수집한다.
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
    # FOMC 연간 캘린더 페이지를 가져온다.
    response = requests.get(CALENDAR_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()

    # 캘린더 HTML을 파싱한다.
    soup = BeautifulSoup(response.text, "html.parser")

    results = []

    # 연도별 패널을 순회하며 "2025 FOMC Meetings" 같은 섹션만 고른다.
    sections = soup.find_all("div", class_="panel-default")

    for section in sections:
        heading = section.find("h4")
        if heading is None:
            continue

        heading_text = heading.get_text(" ", strip=True)
        match = re.match(r"(\d{4}) FOMC Meetings", heading_text)
        if not match:
            continue

        # 각 연도 섹션 안에서 개별 회의 블록을 찾는다.
        meetings = section.find_all("div", class_="fomc-meeting")

        for meeting in meetings:
            # 회의 블록 안의 문서 링크를 순회한다.
            for link in meeting.find_all("a", href=True):
                label = link.get_text(" ", strip=True).lower()
                url = urljoin(BASE_URL, link["href"])

                doc_type = None
                article = None

                # Implementation Note 링크는 전용 파서로 처리한다.
                if "implementation note" in label:
                    doc_type = "implementation_note"
                    article = crawl_implementation_note(url)

                # HTML 링크는 부모 strong 텍스트를 보고
                # statement인지 minutes인지 구분한다.
                elif label == "html":
                    parent_strong = link.parent.find("strong") if link.parent else None
                    if parent_strong is None:
                        continue

                    parent_title = parent_strong.get_text(" ", strip=True).lower()

                    if "statement:" in parent_title:
                        doc_type = "statement"
                        article = crawl_fomc_statement(url)

                    elif "minutes:" in parent_title:
                        doc_type = "minutes"
                        article = crawl_minutes(url)

                        # minutes 공개일은 상세 페이지보다 캘린더 문구가 더 명확해서
                        # 현재 링크가 속한 블록의 Released 문구에서 추출한다.
                        release_match = re.search(
                            r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                            link.parent.get_text(" ", strip=True)
                        )

                        release_date = release_match.group(1) if release_match else None
                        article["release_date"] = release_date

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

                # 연속 요청 부담을 줄이기 위해 짧게 쉰다.
                time.sleep(0.5)

    # 결과를 DataFrame으로 정리하고 중복을 제거한다.
    df = pd.DataFrame(results).drop_duplicates()

    print(df.head(20))

    # 수집 결과를 CSV로 저장한다.
    df.to_csv(collected_csv_path("fed_fomc_links.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

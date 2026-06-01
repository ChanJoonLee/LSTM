import csv
import time
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from pytz import timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path

# 미국 뉴욕 시간대 고정 (서머타임 자동 계산)
NY_TZ = timezone('America/New_York')
# 기본 타겟 데이트: 뉴욕 시간 기준 어제 (YYYY-MM-DD)
TARGET_DATE = (datetime.now(NY_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
TICKERS = ["QQQ", "XLF", "XLE"]


def _save_results(records, target_date):
    csv_path = Path(collected_csv_path(f"yahoo_market_news_{target_date}.csv"))

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sector", "title", "url", "release_date", "body"])
        writer.writeheader()
        writer.writerows(records)

    return csv_path

def convert_to_iso_date(date_str):
    """
    "Mon, June 1, 2026 at 6:18 AM GMT+9" 문자열을 
    실제 미국 뉴욕 시간으로 시차 변환 후 "YYYY-MM-DD"로 반환합니다.
    """
    if not date_str:
        return "N/A"
    try:
        # 1. 'at ' 글자 제거
        cleaned_str = re.sub(r'\s+at\s+', ' ', date_str)
        
        # 2. 타임존 텍스트(GMT+9 등) 분리 및 획득
        # 예: "Mon, June 1, 2026 6:18 AM GMT+9" -> "Mon, June 1, 2026 6:18 AM", "+9"
        match = re.search(r'(GMT)([+-]\d+)$', cleaned_str)
        
        if match:
            tz_offset = match.group(2) # "+9" 확보
            # datetime 파싱을 위해 GMT+9 문구 제거
            cleaned_str = re.sub(r'\s+GMT[+-]\d+$', '', cleaned_str)
        else:
            tz_offset = "+9" # 매칭 실패 시 기본 한국 시간으로 가정
            
        # 3. 일단 텍스트 그대로 datetime 객체 생성 (아직 타임존 정보 없음)
        naive_dt = datetime.strptime(cleaned_str, "%a, %B %d, %Y %I:%M %p")
        
        # 4. 긁어온 원본 시간의 타임존(한국 표준시 KST) 강제 부여
        # 야후 싱가포르나 캐나다 등의 변수를 고려해 offset에 맞게 세팅하는 것이 안전합니다.
        if tz_offset == "+9":
            origin_tz = timezone('Asia/Seoul')
        else:
            origin_tz = timezone('Asia/Seoul') # 예외 시 기본값
            
        localized_dt = origin_tz.localize(naive_dt)
        
        # 5. 🌟 핵심: 실제 미국 뉴욕(증시 기준시) 시간대로 시차 강제 변환
        ny_tz = timezone('America/New_York')
        ny_dt = localized_dt.astimezone(ny_tz)
        
        # 6. 미국 날짜 기준으로 ISO 문자열 출력
        return ny_dt.strftime("%Y-%m-%d")
        
    except Exception as e:
        print(f"[-] 날짜 타임존 변환 실패: {e}")
        return "N/A"

def scrape_news_sync(target_date=TARGET_DATE, tickers=TICKERS):
    print(f"[+] 동기식 Playwright 파이프라인 가동 (기준일자: {target_date})")
    
    clean_dataset = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page = context.new_page()
        
        # 상세 페이지용 공유 탭 하나만 생성
        detail_page = context.new_page()
        
        for ticker in tickers:
            print(f"\n[+] [{ticker}] 섹션 뉴스 목록 로드 중...")
            url = f"https://finance.yahoo.com/quote/{ticker}/news/"
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(2000) 
            
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            
            news_items = soup.select("li.stream-item")
            print(f"[*] 포착된 {ticker} 뉴스 후보: 총 {len(news_items)}개")
            
            # 💡 수정 1: 조기 종료 플래그를 티커 루프 '내부'로 이동시켜 티커 간 간섭 제거
            stop_crawl = False
            
            for item in news_items:
                if stop_crawl:
                    break

                a_tag = item.find("a", class_=["subtitle-link", "titles"]) or item.find("a")
                
                if a_tag and a_tag.get_text():
                    title = a_tag.get_text(strip=True)
                    link = a_tag.get("href", "")
                    
                    if link.startswith("/"):
                        link = "https://finance.yahoo.com" + link
                        
                    if any(kw in title for kw in ["Option", "Put", "Call", "Stock Price", "History"]):
                        continue
                    
                    if "finance.yahoo.com/" not in link:
                        continue
                        
                    print(f"    [-> 상세 페이지 수집] {title[:28]}...")
                    
                    date_str = ""
                    full_body = ""
                    
                    try:
                        detail_page.goto(link, wait_until="domcontentloaded")
                        detail_page.wait_for_timeout(1500) 
                        
                        soup_inner = BeautifulSoup(detail_page.content(), "html.parser")
                        
                        # 상세 날짜 파싱 및 변환
                        time_tag = soup_inner.find("time", class_="byline-attr-meta-time") or soup_inner.find("time")
                        if time_tag:
                            raw_date = time_tag.get_text(strip=True)
                            date_str = convert_to_iso_date(raw_date)

                        # 🛡️ 날짜 조건 체크 및 조기 종료 선언
                        if date_str != "N/A" and date_str < target_date:
                            print(f"    [!] 과거 기사 발견 ({date_str}), {ticker} 수집 조기 종료.")
                            stop_crawl = True
                            break

                        # 상세 본문 파싱
                        body_tag = soup_inner.find("div", class_="bodyItems-wrapper") or soup_inner.find(class_="caas-body")
                        if body_tag:
                            paragraphs = [p.get_text(strip=True) for p in body_tag.find_all("p")]
                            full_body = " ".join(paragraphs)
                        else:
                            time.sleep(1)
                            continue

                    except Exception as detail_err:
                        print(f"    [-] 파싱 에러 패스: {detail_err}")
                        continue 

                    clean_dataset.append({
                        "sector": ticker,
                        "title": title,
                        "url": link,
                        "release_date": date_str if date_str else "N/A",
                        "body": full_body
                    })
                    
                    time.sleep(1)
        
        browser.close()

    return clean_dataset


def main():
    news_results = scrape_news_sync()

    if news_results:
        target_date = news_results[0]["release_date"] if news_results[0].get("release_date") else TARGET_DATE
        output_csv_path = _save_results(news_results, target_date)
        print(
            f"\n[✔] 배치가 완벽히 완료되었습니다. 총 {len(news_results)}개 적재 완료"
            f"\n    - CSV : {output_csv_path}"
        )
    else:
        print("\n[!] 저장할 데이터가 없어 출력 파일 생성을 건너뜁니다.")

if __name__ == "__main__":
    main()
# Data-ML Pipeline

이 저장소는 정책/거시 뉴스 수집부터 뉴스 후처리, 시장 데이터 피처 생성, XGBoost 학습과 비교 실험까지 한 번에 관리하는 데이터 파이프라인 프로젝트.

현재 코드 기준으로는 `shared/` 아래 파이프라인이 메인 실행 경로이고, `training/`과 `crawler/support_legacy/`는 초기 실험 또는 레거시 호환 코드로 분리함.

## 프로젝트 목적(데모)

- 정책/거시 이벤트 문서를 크롤링해 학습 가능한 형태로 정리.
- QQQ와 거시 자산 데이터를 함께 사용해 가격 예측용 피처 생성.
- `market_only`와 `market_news` 두 실험을 같은 절차로 학습해 성능을 비교.

## 현재 기본 설정 요약

아래 내용은 현재 코드 기준의 기본 동작임.

- 메인 뉴스 입력 파일은 `data/crawler/features/merged_finbert_with_embeddings.csv`.
- 이 파일에는 FinBERT 감성 컬럼과 함께 `body_summary_embedding` 컬럼이 있어야 함.
- `shared/news/features.py`는 `body_summary_embedding`을 파싱해 `body_emb_0`, `body_emb_1` 같은 숫자 피처로 펼치고, 행마다 임베딩 차원이 같은지 검증함.
- `market_only`는 고정된 시장 피처 20개를 사용하고, `market_news`는 시장 피처 20개 + 스칼라 뉴스 피처 12개 + `body_emb_*` 30차원을 train 구간에서 `StandardScaler + PCA(5)`로 줄인 임베딩 PC 5개를 사용함.
- 학습용 임베딩 PCA는 처음 기준인 5차원으로 두고, 설명/클러스터용 임베딩은 별도로 PCA(10)를 유지함.
- 뉴스가 없는 날의 일반 뉴스/감성 값은 대부분 0으로 채우되, `body_sentiment_decay_3d`와 `body_emb_*`는 마지막 실제 뉴스의 잔존 효과를 감쇠해서 반영함.
- 클러스터 요약은 비지도 KMeans가 아니라, `market_news` 회귀 모델이 예측한 5거래일 뒤 가격을 현재가와 비교해 예측 수익률을 계산하고, 그 값을 4개 상승/하락 구간으로 나눠 label별 profile을 정리하는 방식임.
- 예측 수익률 profile에도 5일 뉴스 윈도우 평균 임베딩 30차원을 `StandardScaler + PCA(10)`로 줄인 `body_emb_cluster_pc1~10`을 사용함.

## 프로젝트 개요(대략적인 프로젝트 파이프라인)

1. Crawler/collectors/fed.py, crawler/collectors/whitehouse.py, crawler/collectors/bis.py 실행해서 원문 문서 모음
    * 이 단계에서는  FOMC 문서, White House 정책 문서, BIS 보도자료를 CSV 형태로 저장하는것 -> 결과는 data/crawler/collected/ 아래에 쌓임.


2. 수집한 문서를 학습 가능한 형태로 정리하는 단계
    * crawler/postprocessing/text_summarizer.py -> 너무 긴 본문을 Ollama를 이용해서 요약해서 길이를 줄이는 역할
    * crawler/postprocessing/proprocessing.py -> 여러 수집 결과 CSV를 하나로 합치면서 날짜, 카테고리, 문서 타입, 본문 길이 같은 컬럼을 정리
    * crawler/postprocessing/sentiment_score.py -> FinBERT를 사용해 제목과 본문에 감성 점수를 붙여 `merged_finbert.csv` 생성. 메인 학습은 여기에 본문 임베딩이 추가된 `merged_finbert_with_embeddings.csv`를 기본 입력으로 사용
* 이 과정을 거쳐서 모델이 읽을 수 있는 수치화된 뉴스데이터로 변환


3. 학습단계
    * shared/run_market_news_training.py: `merged_finbert_with_embeddings.csv`를 입력으로 받아 전체 파이프라인 실행
    * 맹점은 모델이 문서 한건한건을 직접 읽는 것이 아니라, 하루 단위로 압축된 뉴스 신호를 사용한다는 것.
        * 예를 들어
        * 어떤 날짜에는 뉴스가 몇 건 있었는지
        * 부정 뉴스 비율이 높았는지
        * FOMC 관련 문서가 있었는지
        * 최근 3일과 5일 평균 감성이 어땠는지 같은 값으로 변환한 뒤 시장 데이터와 합친다. 


4. 정리하면
    * 뉴스 피처가 이미 준비된 상태에서 모델 성능만 보고 싶으면 : python shared/run_market_news_training.py만 실행
    * 데이터부터 새로 만들고 싶으면 : collectors -> text_summarizer -> proprocessing -> sentiment_score -> run_market_news_training 순서대로 실행
    * 다만 현재 코드 기준으로 text_summarizer.py는 기본 입력이 BIS 파일 쪽에 맞춰져 있어서, FOMC나 White House 요약까지 자동으로 한 번에 돌리는 구조는 아님. 


## 핵심 실행 흐름

1. `crawler/collectors/`
   외부 사이트에서 원문 문서를 수집.
2. `crawler/postprocessing/`
   긴 문서를 요약하고, 소스별 CSV를 병합하고, 각 소스별 감정 점수 추가.
3. `shared/news/`
   문서 단위 뉴스 데이터를 날짜별 숫자 피처로 집계.
4. `shared/market/`
   QQQ와 거시 자산 가격 데이터를 내려받아 시장 피처를 생성.
5. `shared/training/`
   horizon 선택, 피처 선택, Optuna 튜닝, XGBoost 학습과 평가를 수행.
6. `shared/cluster/`
   `market_news` 테스트 예측값을 4개 예측 수익률 구간으로 나누고, label별 시장/뉴스 profile과 대표 뉴스를 정리.
7. `data/`
   중간 산출물과 최종 모델, 메타데이터, 비교 결과를 저장.

## 디렉터리 구조

```text
data-ml/
├─ crawler/
│  ├─ collectors/
│  │  ├─ fed.py
│  │  ├─ bis.py
│  │  └─ whitehouse.py
│  ├─ postprocessing/
│  │  ├─ text_summarizer.py
│  │  ├─ proprocessing.py
│  │  └─ sentiment_score.py
│  └─ support_legacy/
│     ├─ data_paths.py
│     ├─ pipeline.py
│     ├─ run_crawler.py
│     ├─ scraper.py
│     └─ crawling_test.py
├─ data/
│  ├─ crawler/
│  │  ├─ collected/
│  │  ├─ summarized/
│  │  └─ features/
│  └─ training/
│     ├─ market_only/
│     ├─ market_news/
│     └─ comparison/
├─ shared/
│  ├─ common/
│  ├─ config/
│  ├─ market/
│  ├─ news/
│  ├─ pipelines/
│  ├─ training/
│  └─ run_market_news_training.py
├─ training/
├─ requirements.txt
└─ README.md
```

## 주요 파일 가이드

### 메인 파이프라인

- `shared/run_market_news_training.py`
  가장 먼저 실행하면 되는 메인 CLI 엔트리포인트.(그냥 실행해도 되고, 명령어로 실행해도됨 -> 밑에 설명)
- `shared/pipelines/market_news.py`
  뉴스 로드, 시장 피처 생성, 두 실험 학습, 비교 저장까지의 전체 순서를 관리.
- `shared/training/xgboost_pipeline.py`
  horizon 선택, 피처 선택, Optuna 튜닝, 최종 모델 학습과 평가 수행.

### 뉴스 수집

- `crawler/collectors/fed.py`
  FOMC statement, minutes, implementation note를 수집.
- `crawler/collectors/bis.py`
  BIS 보도자료 목록을 Selenium으로 탐색하고 상세 본문을 수집.
- `crawler/collectors/whitehouse.py`
  White House 문서를 수집한 뒤 QQQ 관련 키워드가 포함된 정책 문서만 남김.

### 뉴스 후처리

- `crawler/postprocessing/text_summarizer.py`
  긴 본문을 Ollama 기반 로컬 LLM으로 요약.
- `crawler/postprocessing/proprocessing.py`
  수집 결과를 표준 컬럼으로 병합하고 카테고리/시간 피처를 추가.
- `crawler/postprocessing/sentiment_score.py`
  FinBERT로 제목/본문 감성 점수를 계산해 최종 뉴스 피처 CSV를 생성.

### 레거시/실험 코드

- `training/train_regression.py`
  초기 단일 회귀 실험 코드.
- `training/dataset.py`
  QQQ 단일 종목 기반 분류 실험 코드.
- `crawler/support_legacy/`
  경로 유틸과 예전 실행 진입점, 간단한 테스트 케이스들을 포함.

## 산출물 저장 위치

### 뉴스 관련

- `data/crawler/collected/`
  크롤러 원문 수집 결과 CSV
- `data/crawler/summarized/`
  요약이 적용된 문서 CSV
- `data/crawler/features/`
  병합, 시간 피처, 감성 점수까지 포함된 학습용 뉴스 CSV

### 학습 관련

- `data/training/market_only/`
  시장 피처만 사용한 실험 결과
- `data/training/market_news/`
  시장 + 뉴스 피처를 사용한 실험 결과
- `data/training/comparison/`
  두 실험의 성능 비교 CSV/JSON

## 코드 실행 가이드

아래 명령은 모두 프로젝트 루트(`data-ml/`)에서 실행하는 것을 기준으로 작성.

### 1. 가상환경 및 기본 패키지 설치(Mac OS 사용시 추천)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 추가 패키지 설치

크롤러/후처리까지 모두 실행하려면 아래 패키지를 추가로 설치해야 함.

```bash
pip install selenium transformers certifi
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

참고:

- `crawler/collectors/bis.py`는 Selenium과 로컬 Chrome/Chromium 환경이 필요.
- `crawler/postprocessing/text_summarizer.py`는 로컬 Ollama 서버가 실행 중이어야 함.
- `crawler/postprocessing/sentiment_score.py`는 `torch`와 `transformers`가 필요함.

## 빠른 실행 시나리오

### A. 이미 뉴스 피처 CSV가 있을 때 학습만 바로 실행

`data/crawler/features/merged_finbert_with_embeddings.csv`가 이미 준비되어 있다면 아래 한 줄로 메인 학습 파이프라인 실행 가능.

```bash
# market_only + market_news + aligned comparison 전체 실행
python shared/run_market_news_training.py

# market_news만 실행 (빠른 반복 실험용, 실행 시간 약 1/3)
python shared/run_market_news_training.py --market-news-only
```

실행이 끝나면 기본적으로 아래 산출물들이 생성됨.(현재 data 폴더안에 생성되어있음)

- `data/training/market_only/qqq_market_only_xgboost_model.json`
- `data/training/market_only/qqq_market_only_metadata.json`
- `data/training/market_news/qqq_market_news_xgboost_model.json`
- `data/training/market_news/qqq_market_news_metadata.json`
- `data/training/comparison/qqq_market_model_comparison.csv`
- `data/training/comparison/qqq_market_model_comparison.json`
- `data/training/comparison/qqq_market_model_comparison_aligned.csv`
- `data/training/comparison/qqq_market_model_comparison_aligned.json`
- `data/training/comparison/qqq_volatility_cluster_model.json`
- `data/training/comparison/qqq_volatility_cluster_report.json`
- `data/training/comparison/qqq_cluster_visualization.png`

### B. 뉴스 수집부터 학습까지 전체 파이프라인 실행

#### Step 1. 뉴스 원문 수집

```bash
python crawler/collectors/fed.py
python crawler/collectors/whitehouse.py
python crawler/collectors/bis.py --max-pages 9
```

기본 출력 위치:

- `data/crawler/collected/fed_fomc_links.csv`
- `data/crawler/collected/whitehouse_qqq_policy.csv`
- `data/crawler/collected/bis_press_releases.csv`

#### Step 2. 긴 문서 요약

```bash
python crawler/postprocessing/text_summarizer.py
```

현재 구현 기준 주의사항:

- `text_summarizer.py`의 기본 `INPUT_CSV`와 `OUTPUT_CSV`는 BIS 파일 기준으로 고정되어 있음.
- `proprocessing.py`는 아래 세 파일이 모두 준비되어 있다고 가정.
  - `data/crawler/summarized/fed_fomc_links_summarized.csv`
  - `data/crawler/summarized/whitehouse_qqq_policy_summarized.csv`
  - `data/crawler/summarized/bis_press_releases_summarized.csv`
- 따라서 FOMC/White House 쪽도 요약 산출물이 필요하면 스크립트 상단 상수를 바꿔 같은 방식으로 다시 실행해야 함.

#### Step 3. 수집 결과 병합 및 시간 피처 생성

```bash
python crawler/postprocessing/proprocessing.py
```

생성 파일:

- `data/crawler/features/merged_table_sorted.csv`
- `data/crawler/features/merged_table_sorted_encoded.csv`
- `data/crawler/features/merged_table_sorted_time_features.csv`

#### Step 4. FinBERT 감성 점수 계산

```bash
python crawler/postprocessing/sentiment_score.py
```

생성 파일:

- `data/crawler/features/merged_finbert.csv`

주의:

- 현재 메인 학습 기본 입력은 `data/crawler/features/merged_finbert_with_embeddings.csv`.
- 이 파일은 `merged_finbert.csv`에 `body_summary_embedding` 컬럼이 추가된 버전이어야 함.
- `body_summary_embedding` 값이 비어 있거나, 파싱되지 않거나, 행마다 차원이 다르면 `shared/news/features.py`에서 바로 오류를 냄.

#### Step 5. 메인 학습 파이프라인 실행

```bash
python shared/run_market_news_training.py
```

## 학습 파이프라인 옵션 예시

기본값 대신 일부 설정을 바꿔 실행할 수도 있음.(종목, 날짜값 등 변경 가능하게)

```bash
python shared/run_market_news_training.py \
  --target-ticker QQQ \
  --start-date 2016-01-01 \
  --end-date 2026-01-01 \
  --horizons 5,10,15,20 \
  --optuna-trials 30 \
  --top-feature-count 20 \
  --training-embedding-pca-components 5
```

주요 옵션:

- `--target-ticker`
  예측 대상 티커
- `--start-date`, `--end-date`
  시장 데이터 다운로드 구간
- `--news-input`
  입력 뉴스 피처 CSV 경로
- `--horizons`
  비교할 예측 horizon 후보
- `--optuna-trials`
  하이퍼파라미터 탐색 횟수
- `--top-feature-count`
  일반 importance 기반 피처 선택 모드에서 남길 상위 피처 개수. 현재 메인 `market_news`의 임베딩 PCA 개수는 아래 `--training-embedding-pca-components`가 결정함
- `--embedding-top-feature-count`
  importance 기반 임베딩 선택 모드에서 사용할 legacy 옵션. 현재 메인 `market_news` 경로는 raw `body_emb_*` 상위 N개 대신 PCA 압축 피처를 사용함
- `--training-embedding-pca-components`
  `market_news` 학습에서 `body_emb_*` 30차원을 몇 개 PCA 축으로 압축할지 결정. 기본값은 5
- `--market-news-only`
  `market_only` 학습과 aligned comparison을 건너뛰고 `market_news`만 실행. 실험 중 빠른 반복이 필요할 때 사용

## 추천 읽기 순서

처음 프로젝트를 파악할 때는 아래 순서로 읽는 것을 추천.

1. `shared/run_market_news_training.py`
2. `shared/pipelines/market_news.py`
3. `shared/market/data.py`
4. `shared/news/features.py`
5. `shared/news/merge.py`
6. `shared/training/xgboost_pipeline.py`

## 현재 코드 기준 메모

- 메인 학습 파이프라인은 `shared/` 아래에 정리.
- `training/` 폴더는 실험, 테스트용 코드로 사용. 새로운 작업은 가급적 `shared/` 기준으로 진행하는 것이 좋을듯.
- 결과 CSV는 `.gitignore`에 의해 기본적으로 Git 추적 대상에서 제외.

## `train_regression.py` 코드가 `shared/`에 반영된 방식

현재 `shared/` 메인 학습은 `training/train_regression.py`를 가능한 한 그 코드 그대로 가져와서 모듈화한 버전으로 보면 됨.

쉽게 말하면:

- `training/train_regression.py`
  한 파일 안에서 데이터 다운로드 -> 피처 생성 -> 뉴스 병합 -> 학습 -> 평가까지 한 번에 처리하는 원본 실험 코드
- `shared/`
  위 흐름을 파일별로 나눠서 유지보수하기 쉽게 만든 구조
  대신 메인 실험 설정은 원본 스크립트와 최대한 같게 맞춰 둠

### 1. 메인 실험 자체를 `train_regression.py`처럼 고정 설정으로 돌림

 메인 실험 기준으로 아래처럼 바뀜.

- `market_only`
  `train_regression.py`에서 쓰는 시장 정예 피처만 사용
- `market_news`
  위 시장 정예 피처 + 뉴스 감성 정예 피처만 사용
- 기본 horizon
  `5일` 고정


참고:

- 이 고정 horizon 값은 `shared/config/schema.py`의 `regression_style_fixed_horizon = 5`
- CLI에서는 `--regression-style-fixed-horizon`으로 바꿀 수 있음

### 2. 시장 피처는 `train_regression.py`의 정예 피처 기준으로 맞춤

`shared/market/data.py`에는 원래 다양한 시장 피처가 많지만, 실제 메인 학습에서 사용하는 피처는 `train_regression.py` 기준 정예 목록으로 제한함.

현재 메인 학습에 쓰는 시장 피처:

- `ret_3`, `ret_5`, `ret_accel`
- `price_to_ma_5`, `slope_5`
- `bb_pos_5`, `bb_width_5`
- `macd_hist`, `rsi_14`
- `vol_5`, `vol_shock`
- `vix_z_score_5`, `drawdown`, `vol_ratio_5`
- `rel_strength_5`
- `uup_ret_5`, `tlt_shock_5`, `hyg_ret_5`
- `target_spy_rel_ret_5`, `target_tlt_rel_ret_5`

특히 아래 계산식은 원본에 맞춰 반영함.

- `ret_accel = (ret_1 / 1.0) - (ret_3 / 3.0)`
- `vol_shock = vol_5 / (vol_20 + 1e-9)`
- `rel_strength_5 = QQQ 5일 수익률 - SPY 5일 수익률`
- `tlt_shock_5 = TLT 5일 수익률`
- `vix_z_score_5 = (VIX - VIX.rolling(5).mean()) / VIX.rolling(5).std()`

즉 `shared` 안에 다른 피처가 더 남아 있더라도, 메인 실험이 실제로 보는 핵심 시장 피처는 `train_regression.py`와 동일한 세트임.

### 3. 뉴스 일자 집계와 임베딩 파싱

`shared/news/features.py`는 `merged_finbert_with_embeddings.csv`를 읽은 뒤:

1. `body_summary_embedding`을 숫자 리스트로 파싱
2. 모든 valid row의 임베딩 차원이 같은지 검증
3. `body_emb_0`, `body_emb_1` 같은 컬럼으로 확장
4. 주말 뉴스를 다음 월요일로 이동
5. 같은 날짜 뉴스는 평균을 내어 하루 1행으로 압축

현재 일자별 뉴스 테이블에 남기는 주요 컬럼:

- `news_count`
- `negative_news_count`, `positive_news_count`
- `negative_news_ratio`, `positive_news_ratio`
- `category_BIS`, `category_FOMC`, `category_UCSB`
- `title_positive_prob`, `title_negative_prob`, `title_neutral_prob`
- `title_sentiment_score`
- `body_positive_prob`, `body_negative_prob`, `body_neutral_prob`
- `body_sentiment_score`
- `body_emb_*`

`day_of_week_*`, `month_*`, `is_weekend`, `body_n_chunks`는 현재 메인 회귀용 뉴스 피처에서는 제외되어 있음.
임베딩 컬럼은 차원이 맞지 않거나 빈 값이 섞이면 조용히 넘어가지 않고 오류를 내도록 방어 코드를 둠.

### 4. 뉴스 병합과 결측 처리 순서도 최대한 그대로 맞춤

`shared/news/merge.py`는 시장 피처와 일자별 뉴스 피처를 날짜 기준으로 합친 뒤 결측을 아래처럼 처리함.

현재 순서:

1. 시장 데이터와 뉴스 일자 테이블을 날짜 기준 `left join`
2. `title_neutral_prob`, `body_neutral_prob`는 기본값 `1.0`으로 채움
3. 일반 뉴스/감성 컬럼은 기본적으로 `0.0`으로 채움 (뉴스 없는 날 = 무신호)
4. `days_since_news` 계산: 마지막 뉴스 이후 경과 거래일 수 (최대 30일)
5. `body_sentiment_decay_3d`, `body_sentiment_decay_5d`, `body_sentiment_decay_7d`, `body_sentiment_decay_15d`는 마지막 실제 뉴스의 `body_sentiment_score`를 가져와 반감기별로 감쇠
6. `body_emb_*`는 마지막 실제 뉴스 임베딩을 가져와 같은 반감기 3일 방식으로 감쇠하되, 최대 5일까지만 반영하고 이후는 0으로 처리
7. 나머지 전체 결측은 `0.0`으로 채움

정리하면:

- 당일 뉴스 자체가 없는 값은 0으로 둠
- 오래된 뉴스의 잔존 영향은 `days_since_news`, 반감기별 `body_sentiment_decay_*d`, 감쇠된 `body_emb_*`로 따로 표현함
- 임베딩은 무한정 `ffill`하지 않고 5일까지만 감쇠 적용함

### 5. 뉴스 파생 피처와 임베딩 선택

현재 메인 `market_news` 실험에서 항상 쓰는 스칼라 뉴스 피처는 아래 12개임.

| 피처 | 설명 |
| --- | --- |
| `news_count_5d` | 최근 5일 뉴스 건수 합산 |
| `days_since_news` | 마지막 뉴스 이후 경과 거래일 수 (최대 30) |
| `sentiment_gap` | 제목 긍정 확률 - 부정 확률 |
| `body_sentiment_gap` | 본문 긍정 확률 - 부정 확률 |
| `sentiment_shock` | `sentiment_gap`의 최근 5일 평균 대비 변화량 |
| `body_sentiment_5d_mean` | 본문 감성 점수 5일 이동평균 |
| `title_sentiment_5d_mean` | 제목 감성 점수 5일 이동평균 |
| `negative_news_spike_5d` | 본문 부정 확률 / 최근 5일 평균 부정 확률 |
| `body_sentiment_decay_3d` | `body_sentiment_score × 0.5^(days_since_news / 3)` — 반감기 3일 감쇠 |
| `fomc_sentiment` | `body_sentiment_score × category_FOMC` |
| `fomc_recent_5d` | 최근 5일 내 FOMC 문서 존재 여부 (rolling max) |
| `sentiment_divergence` | `|title_sentiment_score - body_sentiment_score|` |

`days_since_news`와 `body_sentiment_decay_3d`를 추가한 이유:

- 뉴스가 없는 날 감성값을 `0`으로 채우면 "오늘 뉴스가 있어서 0점"과 "뉴스 자체가 없어서 0점"을 구분 못 함
- `days_since_news`로 경과 일수를 직접 제공하면 모델이 "최근 뉴스"와 "며칠 지난 뉴스"를 구분해서 학습 가능
- `body_sentiment_decay_3d`는 같은 감성 점수라도 오래된 뉴스일수록 영향력이 작아지도록 반감기 감쇠를 적용한 것

추가로 `shared`에서는 aligned comparison 시작일 계산을 위해 `news_count_lag1` 보조 컬럼도 남겨 둠.
이 컬럼은 메인 뉴스 피처라기보다 비교 구간을 자르는 데 쓰는 운영용 컬럼이라고 보면 됨.

학습용 임베딩은 모든 `body_emb_*`를 원본 30차원 그대로 넣지 않고, train 구간에서 PCA로 압축해 넣음.

- 고정 피처: 시장 피처 20개 + 위 스칼라 뉴스 피처 12개
- 임베딩 입력: `body_emb_0~29` 전체 30차원
- 압축 방식: train 구간에서만 `StandardScaler + PCA(5)`를 fit하고, test 구간은 같은 PCA 좌표계로 transform
- 최종 학습용 임베딩 피처: `body_emb_cluster_pc1~5`
- 최종 `market_news` 피처 수: 시장 20개 + 스칼라 뉴스 12개 + 임베딩 PC 5개 = 37개

학습 쪽은 처음 기준인 PCA5로 단순하게 두고, 설명/클러스터 쪽은 대표 뉴스 복원과 해석을 위해 별도의 PCA(10)를 유지함.

train/test 누수를 막기 위해 PCA는 train split에서만 fit하고, test 구간에는 transform만 적용함.

### 6. 학습 타깃과 Optuna 목적함수도 `train_regression.py` 기준

`shared/training/xgboost_pipeline.py`에 반영된 핵심은 아래와 같음.

- 타깃 로그수익률을 `* 100` 스케일로 학습
- 미래 가격 복원 시 `exp(pred_logret / 100.0)` 사용
- Optuna 탐색 범위:
  - `n_estimators: 100 ~ 500`
  - `max_depth: 4 ~ 6`
  - `learning_rate: 0.01 ~ 0.1`
  - `subsample: 0.5 ~ 0.9`
  - `colsample_bytree: 0.5 ~ 0.9`
- 목적함수:
  - `direction_accuracy - rmse * 0.1`

즉 지금 `shared`의 메인 학습은 모델 튜닝 관점에서도 `train_regression.py`와 거의 같은 기준으로 움직임.

### 7. `shared/`에만 남겨둔 구조적 차이

완전히 똑같이 복붙한 것은 아님.
차이는 "실험 구조" 쪽에만 남겨 둔 상태.

- `shared`는 `market_only`와 `market_news`를 같은 실행에서 같이 돌림
- 결과를 `data/training/market_only/`, `market_news/`, `comparison/`에 나눠 저장
- aligned comparison을 따로 만들어 공정 비교를 계속 볼 수 있게 함

`--market-news-only` 플래그를 쓰면 `market_only` 학습과 aligned comparison을 건너뛰고 `market_news` 학습만 돌릴 수 있음.

```bash
python shared/run_market_news_training.py --market-news-only
```

이 모드에서는 실행 시간이 기존의 약 1/3 수준으로 줄어듦.

중요한 점:

- `market_only`는 고정 시장 피처 20개를 사용
- `market_news`는 고정 시장 피처 + 스칼라 뉴스 피처 12개 + train 기준 임베딩 PCA 5개를 사용
- aligned comparison은 `--horizons`에 들어온 후보 horizon들에 대해 같은 선택 규칙으로 다시 비교함

즉 현재 구조를 한 문장으로 정리하면:

- 기본 회귀 학습 로직은 `train_regression.py` 흐름을 따르고
- shared는 그 위에 비교 실험과 저장 구조만 얹어 둔 상태라고 보면 됨.

### 8. 5일 선행 예측 수익률 profile 요약

예측 수익률 profile 요약은 `shared/cluster/model.py`에서 처리함.
현재 방식은 뉴스 피처끼리만 비지도 군집화하는 KMeans가 아니고, 별도 classifier를 새로 학습하는 방식도 아님.
이미 학습된 `market_news` 회귀 모델이 test 구간에서 예측한 `Pred_Future_Price`를 `Current_Price`와 비교해 예측 수익률을 계산하고, 그 예측 수익률을 4개 label로 나눈 뒤 label별 뉴스/시장 profile centroid를 저장함.

기본 설정:

- 예측 대상: `market_news` 회귀 모델이 예측한 `T+5` 가격의 현재가 대비 수익률
- 뉴스 집계 창: anchor 날짜 직전 5일(달력일)
- 레이블 개수: 4개
- label 생성: `(Pred_Future_Price / Current_Price - 1) * 100`
- profile centroid: 모델의 예측 수익률 label별 test 벡터 평균

4개 예측 수익률 레이블 경계:

| 레이블 | 5거래일 뒤 예측 수익률 |
| --- | ---: |
| `fall` | `0%` 미만 |
| `neutral` | `0%` 이상, `+0.3%` 미만 |
| `rise` | `+0.3%` 이상, `+0.6%` 미만 |
| `rise_strong` | `+0.6%` 이상 |

현재 클러스터 피처는 24개임.

기본 시장/뉴스 피처 14개:

- `ret_5`
- `vol_5`
- `vol_ratio_5`
- `drawdown`
- `vix_z_score_5`
- `vol_shock`
- `days_since_news`
- `news_count_lag1`
- `negative_count_ratio_5d`
- `title_sentiment_3d_mean`
- `title_sentiment_5d_mean`
- `body_sentiment_decay_5d`
- `fomc_recent_5d`
- `fomc_sentiment_shock`

임베딩 PCA 피처 10개:

- `body_emb_cluster_pc1~10`

임베딩 PCA 처리 방식:

1. `market_news` 회귀 모델의 test 예측 테이블에서 `Current_Date`, `Current_Price`, `Pred_Future_Price`를 읽음
2. train 구간의 30차원 임베딩 윈도우 벡터에만 `StandardScaler + PCA(n_components=10)`를 fit
3. 전체 이벤트 벡터를 같은 PCA 좌표계로 transform해서 `body_emb_cluster_pc1~10`으로 압축
4. 각 test 예측 row의 `Pred_Future_Price / Current_Price - 1`로 예측 수익률을 계산
5. 예측 수익률을 4개 label로 나누고, 위 14개 기본 피처와 10개 PCA 피처를 합친 24차원 벡터를 label별로 묶음
6. 예측 label별 평균 profile centroid, 평균 예측 수익률, 실제 수익률, 방향성 적중률을 저장

새 뉴스가 들어왔을 때는 PCA를 다시 학습하지 않음.
먼저 저장된 `market_news` 회귀 모델이 미래 가격을 예측하고, 그 예측 가격을 현재가와 비교해 수익률 label을 정함.
`qqq_volatility_cluster_model.json`의 centroid는 그 label이 어떤 뉴스/시장 profile에서 자주 나왔는지를 설명하는 용도임.

`qqq_volatility_cluster_model.json`에는 다음 정보가 함께 저장됨:

- `feature_columns`: 최종 24개 예측 수익률 profile 피처 목록
- `base_feature_columns`: 임베딩 PCA를 제외한 기본 14개 피처 목록
- `embedding_pca.source_columns`: PCA 입력으로 사용한 raw `body_emb_*` 컬럼 목록
- `embedding_pca.feature_columns`: 생성된 `body_emb_cluster_pc*` 컬럼 목록
- `embedding_pca.explained_variance_ratio`: 각 임베딩 PCA 축의 설명분산 비율
- `source_model`: 예측 수익률을 만든 `market_news` 회귀 모델의 horizon과 평가 지표
- `centroids`, `scaler_mean`, `scaler_scale`: test 구간 예측 label별 profile centroid와 시각화용 scaler
- `qqq_volatility_cluster_report.json`의 `predicted_groups[].profile_feature_ranking`:
  label별 centroid가 전체 평균 대비 가장 크게 다른 상위 피처 10개를 `z_diff` 기준으로 저장함
- `qqq_volatility_cluster_report.json`의 `predicted_groups[].representative_embedding_news`:
  label centroid의 `body_emb_cluster_pc*` 값을 원본 30차원 `body_emb_*` 공간으로 복원한 뒤, `merged_finbert_with_embeddings.csv`의 source news 임베딩과 cosine similarity가 높은 실제 뉴스 문서를 저장함

10차원 선택 배경:

- 현재 데이터로 같은 방식의 임베딩 윈도우 PCA를 계산하면 설명분산 합은 5차원 약 24.0%, 10차원 약 44.3%, 15차원 약 61.7%, 20차원 약 77.1%임.
- 5차원은 뉴스 임베딩 정보를 지나치게 버릴 가능성이 있고, 30차원 원본은 profile 요약에서 임베딩 블록이 시장/스칼라 뉴스 피처보다 과하게 커질 수 있음.
- 10차원은 임베딩 정보의 일부를 보존하면서도 원본 30차원 대비 차원을 1/3로 줄이는 절충안이라, 설명/클러스터링 쪽 기본값으로 유지함. 회귀 학습 쪽은 과적합과 피처 수 부담을 줄이기 위해 별도의 PCA(5)를 사용함.

출력 파일:

- `data/training/comparison/qqq_volatility_cluster_model.json`
- `data/training/comparison/qqq_volatility_cluster_report.json`
- `data/training/comparison/qqq_cluster_visualization.png`

`qqq_cluster_visualization.png`를 생성할 때는 label 분리가 잘 보이도록 LDA 2D 투영을 우선 사용하고, 조건이 맞지 않으면 PCA 2D 투영으로 fallback함. 그림 오른쪽에는 각 레이블별로 중심점 피처가 전체 평균 대비 얼마나 차이가 나는지 계산한 상위 피처 순위 테이블을 표시함.

- `+` 값: 해당 피처의 중심값이 전체 이벤트 평균보다 높음
- `-` 값: 해당 피처의 중심값이 전체 이벤트 평균보다 낮음
- `value`: 해당 예측 label profile의 원래 피처 평균값

### 9. 평가 지표 — 고확신 구간 분석 (`evaluate_model`에 추가됨)

`train_regression.py`의 고확신 구간 분석이 `shared/training/xgboost_pipeline.py`의 `evaluate_model` 함수에 이식됨.

**로직:**

1. 테스트셋에서 `|Pred_LogRet|` 기준 상위 30% (70th percentile 이상) 샘플만 추출
2. 상승 예측(Long)과 하락 예측(Short)을 분리해 각각 정확도 계산

```
conf_cutoff = np.quantile(|predicted_logret|, 0.7)
long  → predicted > 0 이고 |predicted| >= conf_cutoff 인 샘플
short → predicted < 0 이고 |predicted| >= conf_cutoff 인 샘플
```

**metrics 딕셔너리에 추가된 키:**

| 키 | 설명 |
| --- | --- |
| `high_conf_threshold` | 70th percentile 기준값 |
| `high_conf_long_accuracy` | Long 예측 정확도 |
| `high_conf_long_count` | Long 샘플 수 |
| `high_conf_short_accuracy` | Short 예측 정확도 |
| `high_conf_short_count` | Short 샘플 수 |

**출력 예시:**

```
Direction accuracy: 56.10%
  [고확신 상위 30%] 기준 문턱값(LogRet 절대값): 0.4505
  상승(Long) 확신 시 정확도: 59.26%  (n=135)
  하락(Short) 확신 시 정확도: 40.00%  (n=5)
```

### 10. 아직 옮기지 않은 부분

아래 요소들은 아직 `shared` 메인 파이프라인에는 넣지 않음.

- importance plot 시각화
- threshold별 전략 곡선
- 모델 평가용 산점도, 누적수익률, 에러 분포 시각화

즉 "모델을 학습하고 비교하는 코어 로직"은 대부분 옮겼고,
"실험 분석용 시각화/리포트 코드"는 아직 `train_regression.py` 쪽에 더 많이 남아 있음.

그래서:

- 최종모델 결과물 생성과 비교는 `shared`에 저장하고
- 모델 수정하면서 진행하는 분석은 `training/train_regression.py`

이렇게 역할을 나눠서 작업해보면 될듯.


## 결과 해석 기준

예전 실험 결과 표는 현재 코드의 기본 설정과 달라져 README에서 제거함. 현재 성능 판단은 매번 새로 생성되는 아래 산출물을 기준으로 보는 것이 가장 안전함.

- `data/training/market_news/qqq_market_news_metadata.json`
- `data/training/market_only/qqq_market_only_metadata.json`
- `data/training/comparison/qqq_market_model_comparison.json`
- `data/training/comparison/qqq_market_model_comparison_aligned.json`
- `data/training/comparison/qqq_volatility_cluster_report.json`

특히 `market_news`와 `market_only` 비교는 다음 조건을 우선 확인해야 함.

- 같은 horizon을 비교했는지
- 같은 날짜 구간을 비교했는지
- 뉴스가 실제로 존재하는 기간만 따로 비교했는지
- `market_news` 학습 피처가 현재 기본값인 시장 20개 + 스칼라 뉴스 12개 + 학습 PCA 5개인지

뉴스 결측 해석에는 여전히 주의가 필요함. 시장 데이터 기간 전체에 뉴스가 촘촘히 존재하는 것은 아니므로, `0`으로 채워진 뉴스 피처가 항상 "그날 실제 뉴스가 없었다"는 뜻은 아닐 수 있음. 장기 성능 판단은 기본 비교보다 aligned comparison 산출물을 우선해서 보는 편이 안전함.

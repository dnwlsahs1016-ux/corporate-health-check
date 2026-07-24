# 기업건강검진 — 코스닥 상장사 고유 재무위험 진단

한국은행 ECOS(거시지표)와 DART(전자공시, 재무제표) Open API 데이터를 이용해 코스닥 상장사의
**거시경제 요인을 제거한 기업 고유(idiosyncratic) 재무위험**을 계산하고, 실제 상장폐지 이력을
기반으로 학습한 모델로 위험점수(0~100)를 보여주는 개인 프로젝트입니다.

> ⚠️ **면책조항**: 이 사이트는 개인 포트폴리오/학습 목적으로 만든 데모입니다. 표시되는
> 위험점수는 소규모 표본으로 학습된 실험적 모델의 결과이며 통계적으로 검증되지 않았습니다.
> **실제 투자 판단, 신용평가, 거래 의사결정에 사용해서는 안 되며**, 특정 기업의 재무 건전성을
> 공식적으로 나타내지 않습니다. 표시된 기업명은 모델 검증을 위한 예시일 뿐입니다.

## 무엇을 보여주나

- 기업별 위험점수(0~100)와 "왜 이 점수가 나왔는지"에 대한 지표별 기여도 설명
- 재무지표 9종의 연도별 추이 — 원지표와 "거시요인을 제거한 고유위험" 비교
- 동종업계(또는 비교기업 전체) 평균 대비 해설 텍스트
- 완전자본잠식처럼 규정상 명확한 위험 신호에 대한 규칙 기반 보정

## 아키텍처

```
DART Open API ──┐
                 ├─▶ 데이터 수집(collect.py) ──▶ 재무비율 계산(features.py)
ECOS Open API ──┘                                     │
                                                        ▼
                                     거시요인 제거(macro_adjust.py, 패널회귀)
                                                        │
                          KIND 상장폐지 이력 ──▶ 라벨링(labels.py)
                                                        │
                                                        ▼
                                  class-weighted 로지스틱 회귀(model.py)
                                                        │
                                                        ▼
                                        위험점수 산출(score.py) + 자본잠식 규칙 보정
                                                        │
                        ┌───────────────────────────────┴───────────────────────────────┐
                        ▼                                                                 ▼
              FastAPI + React (풀스택 데모)                              Streamlit (원클릭 배포용 데모)
```

두 프론트엔드(React / Streamlit) 모두 같은 사전 계산 결과(`backend/data/processed/`)를 읽기만
하므로, 배포된 데모에는 DART/ECOS API 키가 필요 없습니다. 키는 데이터를 새로 수집·재학습할
때만 필요합니다.

## 방법론

### 1. 데이터

- **DART**: `fnlttSinglAcntAll` API로 연결/개별 재무제표(2015~2024) 수집
- **ECOS**: 한국은행 기준금리(722Y001) 시계열
- **KRX KIND**: 코스닥 상장사 전체 목록, 2015년 이후 상장폐지 이력 전체(사유 텍스트 포함)

### 2. 재무지표 (9종)

ROA, ROE, 영업이익률, 부채비율, 유동비율, 자기자본비율, 이자보상배율, 영업현금흐름/총자산,
매출액증가율. 자기자본이 0 이하(완전자본잠식)인 경우 부채비율/ROE의 부호가 뒤집혀 오히려
안전해 보이는 착시가 생기므로, 고정 상하한값으로 대체하고 별도의 `capital_impairment` 플래그로
모델에 노출한다.

### 3. 거시요인 제거

각 재무비율을 `비율 ~ 기준금리 + 산업더미`로 패널회귀(OLS)한 뒤 잔차를 "기업 고유 위험"으로
사용한다. 표본이 작은 산업군에서는 산업더미를 생략하거나 평균차감으로 대체하는 fallback을 둔다.

### 4. 라벨링

KIND 상장폐지 이력(2015~) 중 재무적 사유(감사의견 거절, 자본잠식, 부도, 실질심사 등)로 분류된
건에 한해, **폐지연도-1 회계연도**에 라벨 1을 부여한다(그 해 재무제표로 "내년도 폐지 여부"를
맞추는 문제 설정). 합병·SPAC소멸·자진상장폐지 등은 재무위험과 무관하므로 제외한다.

### 5. 모델

class-weighted 로지스틱 회귀(L2 정규화) + 완전자본잠식 규칙 기반 위험점수 하한(70점) 보정.
로지스틱회귀 계수 × 지표값으로 각 지표의 기여도를 계산해 위험도 산출 근거를 투명하게 보여준다.

## 알려진 한계

- **AUC는 in-sample**: 별도의 연도 기준 학습/검증 분할을 아직 하지 않았다. 지금 수치는 파이프라인
  동작 검증용이며, 실제 예측력에 대한 근거로 보기엔 이르다.
- **일부 지표 계수 부호 불안정**: ROA·자기자본비율처럼 표본 특성상 이론과 반대 부호로 학습된
  지표가 있다. 완전자본잠식처럼 명확한 경우만 규칙으로 보정했다.
- **거시변수 1종**: 기준금리만 사용했다. GDP성장률, 산업생산지수 등으로 확장하면 더 정교해진다.
- **데이터 시점**: 대부분 2024 회계연도 기준이며, 최신 사업보고서 반영은 주기적 재수집이 필요하다.
- **매칭 실패**: 시도한 659개사 중 96개사는 DART corp_code 매칭에 실패해 제외되었다(주로 사명변경/특수문자 기업).

## 로컬 실행

```bash
cd backend
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # 전체 개발(수집+API+모델 재학습)
# 또는
./.venv/Scripts/python.exe -m pip install -r requirements.txt       # Streamlit 데모만 볼 때

cp .env.example .env   # DART_API_KEY, ECOS_API_KEY 입력 (재수집/재학습 시에만 필요)

./.venv/Scripts/python.exe scripts/build_universe.py   # 유니버스 시드 생성
./.venv/Scripts/python.exe scripts/run_pipeline.py      # 수집 → 지표 → 모델 → 위험점수

./.venv/Scripts/python.exe -m streamlit run streamlit_app.py   # Streamlit 데모
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 # FastAPI 서버 (React용)
```

```bash
cd frontend
npm install
npm run dev   # React 프론트엔드 (FastAPI가 떠 있어야 함)
```

## 기술 스택

Python(FastAPI, pandas, statsmodels, scikit-learn) · React(Vite, TypeScript, Recharts) ·
Streamlit(Plotly) · DART/ECOS/KRX KIND Open API

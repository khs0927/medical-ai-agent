# 의료 AI 에이전트

의료 관련 질의응답 및 건강 데이터 분석을 위한 AI 에이전트입니다.

## 주요 기능

- 의학 정보 검색 및 질의응답
- 건강 데이터 분석 및 추세 파악
- RAG(Retrieval-Augmented Generation) 기반 지식 검색
- 환자 기록 요약 및 분석
- 의학 문헌 검색 및 요약
- 임상 가이드라인 기반 권장사항 제공
- 웹 검색 및 스크래핑을 통한 최신 의학 정보 검색

## 설치 방법

1. 저장소 복제:
```bash
git clone https://github.com/yourusername/medical-ai-agent.git
cd medical-ai-agent
```

2. 가상 환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
```bash
cp .env.sample .env
# .env 파일을 열어 필요한 API 키 및 설정 입력
```

## 사용 방법

### 서버 실행

```bash
python run_server.py
```

### 테스트 UI 실행

간단한 웹 기반 테스트 인터페이스를 사용하여 의료 AI 에이전트를 테스트할 수 있습니다:

1. 추가 UI 의존성 설치:
```bash
pip install fastapi uvicorn jinja2
```

2. UI 서버 실행:
```bash
python -m ui.run_ui
```

3. 웹 브라우저에서 `http://localhost:8080`에 접속하여 에이전트와 대화를 시작합니다.

UI에 대한 더 자세한 정보는 [UI README](ui/README.md)를 참조하세요.

### API 엔드포인트

- `/query` - 의학 정보 질의
- `/analyze/health` - 건강 데이터 분석
- `/pubmed/search` - PubMed 검색
- `/semantic/search` - 의미론적 검색

## 웹 검색 기능 활성화

최신 의학 정보에 접근하기 위한 웹 검색 기능:

1. `.env` 파일에서 웹 검색 설정:
```
ENABLE_WEB_SEARCH=true
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

2. Google Custom Search Engine 설정:
   - [Google Cloud Console](https://console.cloud.google.com/)에서 API 키 생성
   - [Google Programmable Search Engine](https://programmablesearchengine.google.com/)에서 커스텀 검색 엔진 생성
   - 의학 관련 사이트 추가 (pubmed.gov, mayoclinic.org 등)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

## 문의

문의사항이 있으시면 [이슈](https://github.com/yourusername/medical-ai-agent/issues)를 생성해주세요. 
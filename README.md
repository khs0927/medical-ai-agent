# 의료 에이전트

의료 상담을 위한 AI 에이전트 시스템입니다.

## 기능

- ECG 데이터 분석
- 위험도 평가
- 약물 상호작용 분석
- 가이드라인 요약
- RAG 기반 의료 지식 검색

## 설치

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/medical-agent.git
cd medical-agent
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키 등을 설정
```

## 실행

```bash
python -m src.medical_agent
```

서버가 http://localhost:8000 에서 실행됩니다.

## API 엔드포인트

- `GET /healthz`: 서버 상태 확인
- `POST /v1/consult`: 의료 상담 요청

### 상담 요청 예시

```bash
curl -X POST "http://localhost:8000/v1/consult" \
     -H "Content-Type: application/json" \
     -d '{"question": "아스피린과 와파린을 같이 복용해도 될까요?"}'
```

## 라이선스

MIT License 
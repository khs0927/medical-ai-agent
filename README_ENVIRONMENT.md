# 환경 변수 설정 가이드

이 문서는 의료 AI 에이전트 프로젝트 실행을 위한 환경 변수 설정 방법을 설명합니다.

## 기본 환경 변수

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 아래 변수를 설정하세요:

```ini
# 로깅 레벨 설정
LOG_LEVEL=INFO

# LLM 제공자 설정
# 'gemini', 'medllama', 'hybrid' 중 선택
LLM_PROVIDER=hybrid
LLM_PRIMARY=gemini

# Gemini API 설정
LLM_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-pro

# 백엔드 API 설정
AI_ANALYZER_API_ENDPOINT=http://localhost:8000
AI_ANALYZER_API_KEY=your_backend_api_key
```

## PostgreSQL 데이터베이스 설정

PostgreSQL 데이터베이스 연결을 위한 환경 변수:

```ini
# PostgreSQL 연결 정보
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=medicaldb
```

## 의료 LLM 모델 설정

MedLLaMA 또는 대체 의료 LLM 모델을 사용하기 위한 설정:

```ini
# MedLLaMA 모델 설정
MEDLLAMA_MODEL_NAME=MedLLaMA-3-8B
MEDLLAMA_MODEL_PATH=medicalai/MedLLaMA-3-8B
MEDLLAMA_DEVICE=cpu  # 'cpu' 또는 'cuda'
MEDLLAMA_USE_QUANTIZATION=True

# Hugging Face 인증 (비공개 모델 접근 시 필요)
HUGGINGFACE_TOKEN=your_huggingface_token

# 공개 의료 모델 대안 (필요 시)
# MEDLLAMA_MODEL_PATH=epfl-llm/meditron-7b
```

## 외부 API 연동 설정

PubMed와 Kaggle API 사용을 위한 환경 변수:

```ini
# PubMed API 설정
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=your_pubmed_api_key
PUBMED_TOOL=MediGenius

# Kaggle API 설정
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Firebase 설정 (레거시)

이전 Firebase를 계속 사용하려는 경우의 설정:

```ini
# Firebase 인증 정보
GOOGLE_APPLICATION_CREDENTIALS=./path/to/firebase-credentials.json
FIREBASE_PROJECT_ID=your-firebase-project-id
```

## 환경 변수 우선순위

1. 코드 내 기본값
2. `.env` 파일 설정
3. 시스템 환경 변수 (OS 수준)

## 변수 구성 예제

### 기본 구성 (Gemini 사용)

```ini
LLM_PROVIDER=gemini
LLM_API_KEY=your_gemini_api_key
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
```

### 하이브리드 구성 (Gemini + 의료 LLM)

```ini
LLM_PROVIDER=hybrid
LLM_PRIMARY=gemini
LLM_API_KEY=your_gemini_api_key
MEDLLAMA_MODEL_PATH=epfl-llm/meditron-7b
MEDLLAMA_DEVICE=cuda
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

### 의료 LLM 전용 구성

```ini
LLM_PROVIDER=medllama
MEDLLAMA_MODEL_PATH=epfl-llm/meditron-7b
MEDLLAMA_DEVICE=cuda
HUGGINGFACE_TOKEN=your_huggingface_token
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
``` 
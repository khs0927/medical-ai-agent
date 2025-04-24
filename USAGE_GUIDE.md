# 의료 AI 에이전트 사용 가이드

이 가이드는 의료 AI 에이전트를 테스트 모드와 프로덕션 모드에서 사용하는 방법을 설명합니다.

## 목차

1. [테스트 모드 (모의 데이터)](#1-테스트-모드-모의-데이터)
2. [프로덕션 모드 (실제 API)](#2-프로덕션-모드-실제-api)
3. [API 키 설정](#3-api-키-설정)
4. [사용 예시](#4-사용-예시)
5. [자주 묻는 질문](#5-자주-묻는-질문)

## 1. 테스트 모드 (모의 데이터)

테스트 모드에서는 실제 API를 호출하지 않고 모의 데이터를 반환합니다. 이는 API 키 없이도 에이전트의 기능을 테스트할 수 있게 해줍니다.

### 1.1. 테스트 서버 실행

```bash
python run_server.py
```

### 1.2. 테스트 클라이언트 사용

```bash
# 대화형 모드
python client.py -i

# 단일 질문 모드
python client.py -q "아스피린과 클로피도그렐 동시 복용해도 되나요?"
```

### 1.3. 자동 테스트 실행

```bash
python test_agent.py
```

## 2. 프로덕션 모드 (실제 API)

프로덕션 모드에서는 실제 API를 호출하여 정확한 응답을 생성합니다. 이를 위해서는 유효한 API 키가 필요합니다.

### 2.1. 프로덕션 서버 실행

```bash
python run_server_production.py
```

### 2.2. 프로덕션 클라이언트 사용

```bash
# 대화형 모드
python client_production.py -i

# 단일 질문 모드
python client_production.py -q "고혈압 환자의 식이요법에 대해 알려주세요."
```

## 3. API 키 설정

프로덕션 모드에서는 다음 API 키가 필요합니다:

### 3.1. `.env.production` 파일 설정

`.env.production` 파일을 열고 다음 정보를 입력하세요:

```
# 프로덕션 환경 변수
GEMINI_API_KEY="your_gemini_api_key"  # 필수
HF_TOKEN="your_huggingface_token"     # 선택 (없으면 일부 기능 모의 데이터 사용)
WEAVIATE_URL="your_weaviate_url"      # 선택 (없으면 모의 검색 결과 사용)
WEAVIATE_API_KEY="your_weaviate_key"  # 선택 (없으면 모의 검색 결과 사용)
TEST_MODE="false"
```

### 3.2. API 키 발급 방법

- **Gemini API 키**: [Google AI Studio](https://makersuite.google.com/app/apikey)에서 발급
- **HuggingFace 토큰**: [HuggingFace Settings](https://huggingface.co/settings/tokens)에서 발급
- **Weaviate**: [Weaviate Cloud Console](https://console.weaviate.cloud/)에서 설정

## 4. 사용 예시

### 4.1. 약물 상호작용 질문

```
질문: 아스피린과 클로피도그렐 동시 복용해도 되나요?
```

### 4.2. 의학 가이드라인 요약

```
질문: STEMI 환자의 응급 처치 가이드라인을 알려주세요.
```

### 4.3. 건강 위험 평가

```
질문: 75세 남성, 흉통, 고혈압, 당뇨 있음. 위험도는 어떤가요?
```

### 4.4. 일반 의학 정보

```
질문: 고혈압 환자의 식이요법에 대해 알려주세요.
```

## 5. 자주 묻는 질문

### Q: 모의 데이터와 실제 API의 차이점은 무엇인가요?
A: 모의 데이터는 미리 정의된 응답을 제공하는 반면, 실제 API는 Gemini와 HuggingFace 모델을 사용하여 더 정확하고 맥락에 맞는 응답을 생성합니다.

### Q: API 키가 없어도 사용할 수 있나요?
A: 테스트 모드에서는 API 키 없이 모의 데이터로 에이전트를 사용할 수 있습니다. 프로덕션 모드에서는 최소한 Gemini API 키가 필요합니다.

### Q: API 호출 비용은 얼마인가요?
A: Gemini API는 사용량에 따라 비용이 발생할 수 있습니다. [Google AI 가격 정책](https://ai.google.dev/pricing)을 참조하세요.

### Q: 에러가 발생하면 어떻게 해야 하나요?
A: 로그를 확인하고 API 키와 네트워크 연결을 확인하세요. 테스트 모드에서 먼저 기능이 정상 작동하는지 테스트해보세요. 
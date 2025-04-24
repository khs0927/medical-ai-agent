# NotToday 앱과 AI 헬퍼 의료 진단 시스템 통합 분석

## 통합 구조 분석

NotToday 앱과 AI 헬퍼 의료 진단 시스템(ADK)은 두 개의 별도 저장소로 유지되면서 API를 통해 통합 운영됩니다. 이 문서는 두 시스템의 통합 방식과 GitHub에서의 연동 방법을 설명합니다.

## 1. 통합 아키텍처

### 1.1 시스템 구성

```
[NotToday 앱(클라이언트)] ←→ [NotToday 서버] ←→ [AI 헬퍼 의료 진단 시스템(ADK)]
```

- **NotToday 앱(클라이언트)**: 사용자 인터페이스와 ECG 데이터 수집 기능 제공
- **NotToday 서버**: 클라이언트와 AI 헬퍼 시스템 사이의 중간 역할
- **AI 헬퍼 의료 진단 시스템(ADK)**: 의료 데이터 분석 및 진단 기능 제공

### 1.2 통신 흐름

1. NotToday 앱에서 ECG 데이터 수집 및 사용자 질의 접수
2. NotToday 서버로 데이터 전송 (API 엔드포인트: `/analysis/consultation`)
3. NotToday 서버가 AI 헬퍼 시스템 호출
4. AI 헬퍼 시스템이 분석 결과 생성
5. 결과가 NotToday 서버를 통해 클라이언트로 반환

## 2. API 명세

### 2.1 NotToday 클라이언트-서버 통신

**요청 (클라이언트 → 서버)**:
```javascript
// POST /api/analysis/consultation
{
  "message": "오늘 심장이 두근거려요. 정상인가요?",
  "userId": "user123",
  "healthData": {
    "heartRate": 72,
    "oxygenLevel": 98,
    "bloodPressure": {
      "systolic": 120,
      "diastolic": 80
    }
  }
}
```

**응답 (서버 → 클라이언트)**:
```javascript
{
  "aiResponse": "심박수 72bpm은 정상 범위(60-100bpm) 내에 있습니다...",
  "timestamp": "2023-08-01T12:34:56Z"
}
```

### 2.2 NotToday 서버-AI 헬퍼 통신

**요청 (서버 → AI 헬퍼)**:
```python
# ADK 시스템 호출
response = MedicalCoordinatorAgent.process(
    query=message,
    context={
        "user_id": userId,
        "health_data": healthData
    }
)
```

## 3. GitHub에서의 통합

두 시스템은 별도의 GitHub 저장소로 관리되지만, 다음과 같은 방법으로 통합이 가능합니다:

### 3.1 저장소 구조

```
github.com/user/NotToday        # NotToday 앱 저장소
github.com/user/medical-agent   # AI 헬퍼 의료 진단 시스템 저장소
```

### 3.2 통합 옵션

다음 세 가지 방법 중 하나로 통합할 수 있습니다:

#### 옵션 1: Git Submodules (권장)

NotToday 저장소가 AI 헬퍼 시스템을 Git Submodule로 참조하는 방식:

```bash
# NotToday 저장소 내에서
git submodule add https://github.com/user/medical-agent.git ai-helper
git commit -m "Add medical agent as submodule"
```

이 방식은 두 시스템의 코드를 독립적으로 관리하면서 특정 버전으로 고정할 수 있습니다.

#### 옵션 2: Docker 컨테이너 통합

AI 헬퍼 시스템을 Docker 컨테이너로 배포하고 NotToday 서버에서 API로 호출:

```yaml
# docker-compose.yml
services:
  nottoday-server:
    image: nottoday-server:latest
    ports:
      - "8000:8000"
  
  ai-helper:
    image: medical-agent:latest
    ports:
      - "8080:8080"
```

#### 옵션 3: 독립적 배포 및 API 통합

두 시스템을 완전히 별도로 배포하고 API를 통해 통신하는 방식:

1. AI 헬퍼 시스템을 별도 서버에 배포
2. NotToday의 환경 설정에 AI 헬퍼 API 엔드포인트 지정
3. NotToday 서버가 HTTP/HTTPS를 통해 원격 AI 헬퍼 시스템 호출

## 4. 환경 설정

### 4.1 NotToday 환경 변수

```
# NotToday/.env
AI_HELPER_ENDPOINT=http://localhost:8080
AI_HELPER_API_KEY=your_api_key_here
```

### 4.2 AI 헬퍼 환경 변수

```
# medical-agent/.env
HUGGINGFACE_TOKEN=your_hf_token_here
SERVING_PORT=8080
```

## 5. 배포 시나리오 분석

### 5.1 단일 서버 배포

소규모 배포에 적합:
- NotToday 서버와 AI 헬퍼 시스템을 동일 서버에 배포
- Docker Compose로 컨테이너 관리
- 로컬 네트워크 통신으로 지연 시간 최소화

### 5.2 분산 서버 배포

대규모 배포에 적합:
- NotToday 서버: 웹 프론트엔드와 API 처리
- AI 헬퍼 시스템: 별도의 고성능 서버에서 의료 분석 처리
- 로드 밸런서로 여러 AI 헬퍼 인스턴스 관리 가능

## 6. 결론

NotToday 앱과 AI 헬퍼 의료 진단 시스템은 GitHub에서 별도의 저장소로 관리되지만, 효과적으로 통합 가능합니다. Git Submodule 방식이 코드 관리와 버전 호환성 측면에서 가장 권장됩니다. API 기반 통신으로 두 시스템이 효과적으로 상호작용하며, 환경변수를 통해 쉽게 구성할 수 있습니다.

현재 NotToday의 `/api/analysis/consultation` 엔드포인트가 AI 헬퍼 시스템과 통신하도록 이미 변경되었으므로, 통합 과정에서 추가적인 코드 수정은 최소화할 수 있습니다.

## 7. 다음 단계

1. AI 헬퍼 의료 진단 시스템 GitHub 저장소 생성
2. NotToday 저장소에 AI 헬퍼 시스템을 서브모듈로 추가
3. 환경 변수 설정 및 통합 테스트
4. 필요한 경우 API 엔드포인트 조정 
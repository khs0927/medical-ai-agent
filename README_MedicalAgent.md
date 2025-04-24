# AI 헬퍼 의료 진단 시스템

[![한국어](https://img.shields.io/badge/한국어-주요언어-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Google ADK](https://img.shields.io/badge/Google-ADK-green)](https://github.com/google/adk-python)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-API-yellow)](https://huggingface.co/inference-api)

AI 헬퍼 의료 진단 시스템은 Google의 Agent Development Kit(ADK)와 Hugging Face의 의료 특화 모델을 기반으로 하는 심혈관 진단 및 건강 상담 AI 에이전트입니다. 사용자의 건강 데이터를 분석하고 전문적인 의료 정보를 제공합니다.

<p align="center">
  <img src="assets/agent-development-kit.png" width="200" />
</p>

## 🌟 특징

- **심혈관 건강 데이터 분석**: ECG 데이터, 심박수, 혈압 등의 생체 데이터를 분석
- **개인화된 건강 위험 평가**: 사용자 정보와 건강 지표를 기반으로 심혈관 위험 평가
- **전문적인 건강 상담**: 건강 질문에 대한 정확하고 신뢰할 수 있는 응답 제공
- **사용하기 쉬운 인터페이스**: Google ADK의 웹 인터페이스를 통한 간편한 상호작용
- **PubMed/Kaggle 통합**: 최신 의학 연구 및 데이터셋 검색 및 참조 기능
- **안전하고 책임감 있는 AI**: 의학적 정확성을 우선시하고 명확한 면책 조항 제공

## 🔧 설치 및 설정

### 요구사항

- Python 3.9 이상
- [Google ADK](https://github.com/google/adk-python) 
- Hugging Face API 키

### 설치 방법

1. 저장소 복제

```bash
git clone https://github.com/your-username/medical-agent-system.git
cd medical-agent-system
```

2. 가상 환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치

```bash
pip install -r requirements.txt
```

4. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키 설정
```

## 📚 사용 방법

### 기본 사용법

1. ADK 개발 UI 실행:

```bash
adk dev ./src/medical_agent
```

2. 웹 브라우저에서 http://localhost:8080 접속

3. 의료 상담 시작!

### 명령줄에서 사용:

```bash
python -m agent "What are the symptoms of a heart attack?"
```

### 프로그래밍 방식으로 사용:

```python
from medical_agent.agents import MedicalCoordinatorAgent

# 에이전트 초기화
agent = MedicalCoordinatorAgent

# 쿼리 처리
response = agent.process("What should I do to improve my heart health?")
print(response)
```

## 🔄 NotToday 앱과의 통합

AI 헬퍼 의료 진단 시스템은 NotToday 앱과 완벽하게 통합되어 작동합니다.

### 통합 방법

1. NotToday 서버 설정:
   - `/analysis/consultation` 엔드포인트를 통해 AI 헬퍼 의료 시스템에 연결
   - `.env` 파일에 올바른 API 키 및 엔드포인트 구성

2. NotToday 클라이언트 설정:
   - 클라이언트의 `/api/analysis/consultation` 호출이 서버의 AI 헬퍼 엔드포인트로 라우팅되도록 설정

### 예시 코드

**NotToday의 서버 라우트 설정 (Node.js/Express):**

```javascript
// NotToday/server/routes.ts
import { Router } from 'express';
import { AnalysisController } from '../controllers/AnalysisController';

const router = Router();

// AI 헬퍼 상담 엔드포인트
router.post('/analysis/consultation', AnalysisController.handleAIConsultation);

export default router;
```

**AI 헬퍼 에이전트 호출 (서버 측):**

```javascript
// NotToday/server/controllers/AnalysisController.ts
import { MedicalAgentClient } from '../services/MedicalAgentClient';

export class AnalysisController {
  static async handleAIConsultation(req, res) {
    try {
      const { message, userId } = req.body;
      
      // AI 헬퍼 에이전트 호출
      const response = await MedicalAgentClient.processMedicalQuery(message, userId);
      
      res.json({ aiResponse: response });
    } catch (error) {
      console.error('AI 헬퍼 상담 오류:', error);
      res.status(500).json({ error: 'AI 상담 처리 중 오류가 발생했습니다.' });
    }
  }
}
```

## 🧠 시스템 아키텍처

AI 헬퍼 의료 진단 시스템은 다음과 같은 구성 요소로 이루어져 있습니다:

1. **MedicalCoordinatorAgent**: 중앙 에이전트로, 사용자 요청을 이해하고 적절한 도구를 호출
2. **분석 도구**:
   - ECG 데이터 분석 도구
   - 건강 위험 평가 도구
3. **HuggingFaceClient**: Hugging Face 모델 API와 통신하여 고급 의학 분석 수행
4. **프롬프트 엔진**: 전문화된 의료 프롬프트로 정확하고 상세한 분석 결과 생성

## 🛠️ 개발자 정보

### 프로젝트 구조

```
medical-agent-system/
├── src/
│   ├── medical_agent/
│   │   ├── __init__.py
│   │   ├── agents.py     # 주요 에이전트 정의
│   │   ├── tools.py      # 도구 및 분석 함수
│   │   ├── prompts.py    # 의학 분석용 프롬프트
│   │   └── hf_client.py  # Hugging Face API 클라이언트
├── agent.py              # 메인 에이전트 실행 스크립트
├── .env.example          # 환경 변수 예시
└── README.md             # 이 문서
```

### 확장하기

1. 새로운 도구 추가:
   - `tools.py`에 새로운 함수 추가 및 ADK 도구로 데코레이션
   - `agents.py`에서 에이전트 도구 목록에 추가

2. 새로운 분석 기능:
   - `prompts.py`에 특화된 프롬프트 추가
   - `tools.py`에 해당 분석 로직 구현

## 📜 라이선스

이 프로젝트는 Apache License 2.0 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## ⚠️ 면책 조항

이 시스템은 교육 및 정보 제공 목적으로만 사용됩니다. 심각한 건강 문제가 있는 경우 항상 의료 전문가와 상담하세요. AI가 생성한 정보는 실제 의학적 조언을 대체할 수 없습니다. 
# Gemini 2.5 Pro Preview 통합 및 GitHub 배포 안내

이 문서는 Gemini 2.5 Pro Preview 모델을 활용하는 의료 진단 에이전트를 GitHub에 배포하는 방법을 안내합니다.

## 현재 상태

- LLM 모델을 `gemini-1.5-flash`에서 `gemini-2.5-pro-preview-03-25`로 업데이트했습니다.
- 로컬 Git 저장소에 변경 사항을 커밋했습니다.
- GitHub 리포지토리 연동 준비가 완료되었습니다.
- API 키 설정이 필요합니다(현재 테스트 실행 시 API 키 오류 발생).

## API 키 설정 방법

실제 배포 환경에서는 다음과 같이 환경 변수를 설정해야 합니다:

1. **API 키 발급**
   - Google AI Studio(https://aistudio.google.com/)에 접속합니다.
   - Google 계정으로 로그인합니다.
   - "Get API key" 섹션으로 이동하여 새 API 키를 생성합니다.
   - 생성된 키를 안전하게 보관합니다.

2. **로컬 개발 환경**
   - `.env` 파일에 유효한 API 키를 입력합니다.
   ```
   LLM_API_KEY="유효한_GEMINI_API_키"
   ```

3. **GitHub Actions를 통한 배포(선택사항)**
   - GitHub 리포지토리의 Settings > Secrets > Actions에서 새 시크릿을 추가합니다.
   - `LLM_API_KEY`라는 이름으로 API 키 값을 저장합니다.
   - 워크플로우 파일에서 환경 변수로 참조합니다.

4. **Vercel/Netlify 등 호스팅 서비스(선택사항)**
   - 해당 호스팅 서비스의 환경 변수 설정에서 API 키를 설정합니다.

## GitHub 배포 단계

1. **GitHub 리포지토리 생성**
   - GitHub 계정에 로그인합니다.
   - "New repository" 버튼을 클릭합니다.
   - 리포지토리 이름을 지정하고(예: "medical-ai-agent") 필요한 설정을 완료합니다.
   - "Create repository" 버튼을 클릭합니다.

2. **원격 저장소 연결**
   ```bash
   git remote add origin https://github.com/사용자명/medical-ai-agent.git
   ```

3. **변경 사항 푸시**
   ```bash
   git push -u origin main
   ```

## Gemini 2.5 Pro Preview 모델 특징

이번 업데이트에서 `gemini-2.5-pro-preview-03-25` 모델을 사용함으로써 다음과 같은 이점이 있습니다:

- 코딩, 추론, 멀티모달 이해에 최적화된 성능
- 복잡한 의료 문제에 대한 더 정확한, 근거 기반 추론
- 어려운 의학적 코딩, 수학, STEM 문제 해결 능력 향상
- 대규모 의료 데이터셋, 환자 기록, 의학 문헌 분석을 위한 향상된 컨텍스트 처리

## 주의사항

- API 키는 항상 안전하게 관리하고, 소스 코드에 직접 포함하지 마세요.
- 실제 의료 환경에서 사용하기 전에 충분한 테스트를 수행하세요.
- Gemini 2.5 Pro Preview는 최신 모델로, API 변경사항이 있을 수 있으므로 Google AI 공식 문서를 주기적으로 확인하세요. 
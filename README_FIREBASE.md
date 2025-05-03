# Firebase 백엔드 설정 가이드

이 문서는 의료 AI 에이전트 프로젝트에서 Firebase 백엔드를 설정하고 사용하는 방법을 설명합니다.

## 사전 요구사항

- Python 3.8 이상
- Firebase 프로젝트 생성 (https://console.firebase.google.com/)
- Firebase 서비스 계정 키 (JSON 파일)

## 설치

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. Firebase 서비스 계정 키 준비
   - Firebase 콘솔 → 프로젝트 설정 → 서비스 계정 → "새 비공개 키 생성" 클릭
   - 다운로드된 JSON 파일을 안전한 위치에 저장

## 환경 변수 설정

`.env` 파일을 생성하고 다음 변수를 설정하세요:

```
# Firebase 설정
FIREBASE_PROJECT_ID=your-project-id

# 다음 중 하나 사용:
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/serviceAccountKey.json
# 또는
# FIREBASE_CREDENTIALS={"type":"service_account","project_id":"...","private_key":"...","client_email":"..."}

# LLM API 설정
LLM_PROVIDER=gemini
LLM_API_KEY=your-gemini-api-key
```

## Firestore 데이터베이스 구조

Firebase 백엔드는 다음과 같은 컬렉션을 사용합니다:

1. `medical_literature` - 의학 문헌 정보
   - 문서 ID: 자동 생성 또는 DOI/PubMed ID
   - 필드: title, authors, publication_date, journal, abstract, full_text, doi, mesh_terms 등

2. `clinical_guidelines` - 임상 가이드라인
   - 문서 ID: 자동 생성
   - 필드: title, organization, publish_date, update_date, specialty, recommendation_level, content 등

3. `patient_records` - 환자 기록
   - 문서 ID: 환자 ID (문자열)
   - 필드: demographics, medical_history, medications, lab_results, vitals 등

4. `medical_images` - 의료 이미지 메타데이터
   - 문서 ID: 자동 생성
   - 필드: patient_id, study_id, modality, body_part, acquisition_date, metadata, storage_path 등

## 보안 규칙 설정

Firestore 보안 규칙을 다음과 같이 설정하는 것을 권장합니다:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // 인증된 사용자만 접근 가능
    match /{document=**} {
      allow read, write: if request.auth != null;
    }
    
    // 환자 데이터는 특별한 권한이 필요
    match /patient_records/{patientId} {
      allow read, write: if request.auth != null && 
        (request.auth.token.role == 'admin' || request.auth.token.role == 'doctor');
    }
  }
}
```

## 파일 저장소 설정 (선택 사항)

의료 이미지 또는 큰 문서를 저장하려면 Firebase Storage를 설정하세요:

1. Firebase 콘솔 → Storage → 시작하기
2. 보안 규칙 설정
3. 다음과 같이 코드에서 사용:

```python
from firebase_admin import storage

bucket = storage.bucket()
blob = bucket.blob('medical_images/patient123/xray.jpg')
blob.upload_from_filename('local_path/to/xray.jpg')
```

## 서버 실행

Firebase 백엔드로 서버를 실행하려면:

```bash
python run_server.py
```

## 알려진 제한 사항

1. Firestore는 부분 텍스트 검색을 직접 지원하지 않습니다. 복잡한 검색 기능이 필요한 경우 Algolia나 Elasticsearch와 같은 전용 검색 서비스 통합을 고려하세요.

2. 벡터 검색을 위해서는 외부 벡터 데이터베이스(Pinecone, Weaviate 등)와의 통합이 필요할 수 있습니다.

## 추가 리소스

- [Firebase 공식 문서](https://firebase.google.com/docs)
- [Firebase Admin SDK for Python](https://firebase.google.com/docs/admin/setup)
- [Firestore 문서](https://firebase.google.com/docs/firestore) 
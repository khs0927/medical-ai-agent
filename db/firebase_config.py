"""
Firebase 데이터베이스 구성 및 연결 관리
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# Firebase 관련 환경 변수
FIREBASE_CREDENTIALS = os.getenv('FIREBASE_CREDENTIALS')
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')

# 전역 변수 (싱글톤 패턴)
_firebase_app = None
_firestore_client = None

def initialize_firebase():
    """Firebase 앱 초기화"""
    global _firebase_app, _firestore_client
    
    # 이미 초기화된 경우 건너뜀
    if _firebase_app:
        return _firestore_client
    
    try:
        # 서비스 계정 키 파일 경로 확인
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # 환경 변수에서 자격 증명 JSON 사용 (선호)
        if FIREBASE_CREDENTIALS:
            try:
                cred_dict = json.loads(FIREBASE_CREDENTIALS)
                cred = credentials.Certificate(cred_dict)
            except json.JSONDecodeError:
                logger.error('FIREBASE_CREDENTIALS 환경 변수가 유효한 JSON이 아닙니다.')
                raise
        # 파일 경로 사용
        elif cred_path and Path(cred_path).exists():
            cred = credentials.Certificate(cred_path)
        # 기본 자격 증명 사용 (GCP 환경)
        else:
            # GCP에서 실행 중일 때 자격 증명이 자동으로 감지됨
            cred = None
            logger.info('기본 GCP 자격 증명을 사용하여 Firebase 초기화 시도')
        
        # Firebase 앱 초기화
        _firebase_app = firebase_admin.initialize_app(cred)
        
        # Firestore 클라이언트 초기화
        _firestore_client = firestore.client()
        
        logger.info('Firebase 및 Firestore 초기화 완료')
        return _firestore_client
    
    except Exception as e:
        logger.error(f'Firebase 초기화 오류: {e}', exc_info=True)
        raise

def get_firestore_client():
    """Firestore 클라이언트 가져오기 (필요시 초기화)"""
    global _firestore_client
    
    if not _firestore_client:
        _firestore_client = initialize_firebase()
        
    return _firestore_client

# 컬렉션 이름 상수
COLLECTION_MEDICAL_LITERATURE = 'medical_literature'
COLLECTION_CLINICAL_GUIDELINES = 'clinical_guidelines'
COLLECTION_PATIENT_RECORDS = 'patient_records'
COLLECTION_MEDICAL_IMAGES = 'medical_images'
COLLECTION_EMBEDDINGS = 'embeddings' 
"""
PostgreSQL 데이터베이스 구성 및 연결 관리
"""
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# PostgreSQL 연결 정보
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'medicaldb')

# SQLAlchemy 연결 문자열
DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# 전역 변수 (싱글톤 패턴)
_engine = None
_SessionLocal = None
Base = declarative_base()

def initialize_db():
    """데이터베이스 엔진 및 세션 팩토리 초기화"""
    global _engine, _SessionLocal
    
    # 이미 초기화된 경우 건너뜀
    if _engine:
        return _SessionLocal
    
    try:
        # 데이터베이스 엔진 생성
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # 30분마다 연결 갱신
            pool_pre_ping=True  # 연결 유효성 검사
        )
        
        # 세션 팩토리 생성
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        
        logger.info('PostgreSQL 데이터베이스 연결 초기화 완료')
        return _SessionLocal
    
    except Exception as e:
        logger.error(f'PostgreSQL 데이터베이스 초기화 오류: {e}', exc_info=True)
        raise

def get_db_session():
    """데이터베이스 세션 가져오기"""
    if not _SessionLocal:
        initialize_db()
    
    db = _SessionLocal()
    try:
        return db
    finally:
        db.close()

# 테이블 이름 상수
TABLE_MEDICAL_LITERATURE = 'medical_literature'
TABLE_CLINICAL_GUIDELINES = 'clinical_guidelines'
TABLE_PATIENT_RECORDS = 'patient_records'
TABLE_MEDICAL_IMAGES = 'medical_images'
TABLE_EMBEDDINGS = 'embeddings' 
"""
PostgreSQL 데이터베이스 모델 정의
"""
import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import relationship

from db.postgres_config import Base
from db.postgres_config import TABLE_CLINICAL_GUIDELINES
from db.postgres_config import TABLE_MEDICAL_LITERATURE
from db.postgres_config import TABLE_PATIENT_RECORDS


class MedicalLiterature(Base):
    """의학 문헌 모델"""
    __tablename__ = TABLE_MEDICAL_LITERATURE

    id = Column(String(50), primary_key=True)
    title = Column(String(255), nullable=False, index=True)
    authors = Column(JSON, nullable=True)
    publication_date = Column(String(50), nullable=True)
    journal = Column(String(255), nullable=True)
    abstract = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    doi = Column(String(100), nullable=True, unique=True)
    mesh_terms = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # 시간 정보
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 임베딩 관계
    embedding_id = Column(String(50), ForeignKey('embeddings.id'), nullable=True)
    embedding = relationship('Embedding', back_populates='medical_literature')

class ClinicalGuideline(Base):
    """임상 가이드라인 모델"""
    __tablename__ = TABLE_CLINICAL_GUIDELINES

    id = Column(String(50), primary_key=True)
    title = Column(String(255), nullable=False, index=True)
    organization = Column(String(255), nullable=True)
    publish_date = Column(String(50), nullable=True)
    update_date = Column(String(50), nullable=True)
    specialty = Column(String(100), nullable=True)
    recommendation_level = Column(String(10), nullable=True)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)
    
    # 시간 정보
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 임베딩 관계
    embedding_id = Column(String(50), ForeignKey('embeddings.id'), nullable=True)
    embedding = relationship('Embedding', back_populates='clinical_guideline')

class PatientRecord(Base):
    """환자 기록 모델"""
    __tablename__ = TABLE_PATIENT_RECORDS

    id = Column(String(50), primary_key=True)  # 환자 ID
    demographics = Column(JSON, nullable=False)  # 인구통계학적 정보 (나이, 성별, 키, 체중 등)
    medical_history = Column(JSON, nullable=True)  # 병력 (조건, 알레르기, 수술 이력 등)
    medications = Column(JSON, nullable=True)  # 약물 (이름, 용량, 빈도 등)
    lab_results = Column(JSON, nullable=True)  # 검사 결과
    vitals = Column(JSON, nullable=True)  # 활력 징후 (심박수, 혈압 등)
    
    # 시간 정보
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Embedding(Base):
    """텍스트 임베딩 모델"""
    __tablename__ = 'embeddings'

    id = Column(String(50), primary_key=True)
    vector = Column(JSON, nullable=False)  # 임베딩 벡터
    model_name = Column(String(100), nullable=False)  # 임베딩 모델 이름
    document_type = Column(String(50), nullable=False)  # 문서 유형 (medical_literature, clinical_guideline 등)
    
    # 관계
    medical_literature = relationship('MedicalLiterature', back_populates='embedding', uselist=False)
    clinical_guideline = relationship('ClinicalGuideline', back_populates='embedding', uselist=False)
    
    # 시간 정보
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow) 
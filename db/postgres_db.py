"""
PostgreSQL 데이터베이스 어댑터
"""
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import uuid

from sqlalchemy import text

from db.postgres_config import get_db_session
from db.postgres_models import ClinicalGuideline
from db.postgres_models import MedicalLiterature
from db.postgres_models import PatientRecord

# 로깅 설정
logger = logging.getLogger(__name__)


class PostgresDB:
    """PostgreSQL을 사용하는 데이터베이스 어댑터"""

    def __init__(self):
        """PostgreSQL 세션 초기화"""
        self.get_db = get_db_session
        logger.info('PostgresDB 인스턴스 생성 완료')

    async def add_document(self, collection: str, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """PostgreSQL에 문서 추가

        Args:
            collection: 테이블 이름
            data: 저장할 데이터
            doc_id: 사용할 문서 ID (없으면 자동 생성)

        Returns:
            생성된 문서 ID
        """
        try:
            # 문서 ID 생성 또는 사용
            if not doc_id:
                doc_id = str(uuid.uuid4())
            
            # 데이터에 ID 포함
            data['id'] = doc_id
            
            # 세션 가져오기
            db = self.get_db()
            
            try:
                # 컬렉션에 따라 다른 모델 사용
                if collection == 'medical_literature':
                    item = MedicalLiterature(**data)
                elif collection == 'clinical_guidelines':
                    item = ClinicalGuideline(**data)
                elif collection == 'patient_records':
                    item = PatientRecord(**data)
                else:
                    raise ValueError(f'지원되지 않는 컬렉션: {collection}')
                
                # 데이터베이스에 추가
                db.add(item)
                db.commit()
                
                logger.info(f'문서가 {collection}/{doc_id}에 추가되었습니다.')
                return doc_id
            
            except Exception as e:
                db.rollback()
                logger.error(f'{collection} 테이블에 문서 추가 중 오류 발생: {e}', exc_info=True)
                raise
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'{collection} 컬렉션에 문서 추가 중 오류 발생: {e}', exc_info=True)
            raise

    async def get_document(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """PostgreSQL에서 특정 문서 가져오기

        Args:
            collection: 테이블 이름
            doc_id: 문서 ID

        Returns:
            문서 데이터 또는 None (존재하지 않는 경우)
        """
        try:
            # 세션 가져오기
            db = self.get_db()
            
            try:
                # 컬렉션에 따라 다른 모델 사용
                if collection == 'medical_literature':
                    model = MedicalLiterature
                elif collection == 'clinical_guidelines':
                    model = ClinicalGuideline
                elif collection == 'patient_records':
                    model = PatientRecord
                else:
                    raise ValueError(f'지원되지 않는 컬렉션: {collection}')
                
                # 문서 조회
                item = db.query(model).filter(model.id == doc_id).first()
                
                if item:
                    # SQLAlchemy 모델을 딕셔너리로 변환
                    result = self._model_to_dict(item)
                    return result
                else:
                    logger.warning(f'{collection}/{doc_id} 문서가 존재하지 않습니다.')
                    return None
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'{collection}/{doc_id} 문서 조회 중 오류 발생: {e}', exc_info=True)
            raise

    async def update_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> bool:
        """PostgreSQL 문서 업데이트

        Args:
            collection: 테이블 이름
            doc_id: 문서 ID
            data: 업데이트할 데이터

        Returns:
            성공 여부
        """
        try:
            # 세션 가져오기
            db = self.get_db()
            
            try:
                # 컬렉션에 따라 다른 모델 사용
                if collection == 'medical_literature':
                    model = MedicalLiterature
                elif collection == 'clinical_guidelines':
                    model = ClinicalGuideline
                elif collection == 'patient_records':
                    model = PatientRecord
                else:
                    raise ValueError(f'지원되지 않는 컬렉션: {collection}')
                
                # 문서 조회
                item = db.query(model).filter(model.id == doc_id).first()
                
                if not item:
                    logger.warning(f'{collection}/{doc_id} 문서가 존재하지 않습니다.')
                    return False
                
                # 각 필드 업데이트
                for key, value in data.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                
                # 업데이트 실행
                db.commit()
                
                logger.info(f'{collection}/{doc_id} 문서가 업데이트되었습니다.')
                return True
            
            except Exception as e:
                db.rollback()
                logger.error(f'{collection}/{doc_id} 문서 업데이트 중 오류 발생: {e}', exc_info=True)
                return False
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'{collection}/{doc_id} 문서 업데이트 중 오류 발생: {e}', exc_info=True)
            return False

    async def delete_document(self, collection: str, doc_id: str) -> bool:
        """PostgreSQL 문서 삭제

        Args:
            collection: 테이블 이름
            doc_id: 문서 ID

        Returns:
            성공 여부
        """
        try:
            # 세션 가져오기
            db = self.get_db()
            
            try:
                # 컬렉션에 따라 다른 모델 사용
                if collection == 'medical_literature':
                    model = MedicalLiterature
                elif collection == 'clinical_guidelines':
                    model = ClinicalGuideline
                elif collection == 'patient_records':
                    model = PatientRecord
                else:
                    raise ValueError(f'지원되지 않는 컬렉션: {collection}')
                
                # 문서 삭제
                item = db.query(model).filter(model.id == doc_id).first()
                
                if not item:
                    logger.warning(f'{collection}/{doc_id} 문서가 존재하지 않습니다.')
                    return False
                
                # 삭제 실행
                db.delete(item)
                db.commit()
                
                logger.info(f'{collection}/{doc_id} 문서가 삭제되었습니다.')
                return True
            
            except Exception as e:
                db.rollback()
                logger.error(f'{collection}/{doc_id} 문서 삭제 중 오류 발생: {e}', exc_info=True)
                return False
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'{collection}/{doc_id} 문서 삭제 중 오류 발생: {e}', exc_info=True)
            return False

    async def query_documents(
        self, collection: str, filters: Optional[List[Tuple[str, str, Any]]] = None, 
        limit: int = 10, order_by: Optional[str] = None, order_direction: str = 'DESC'
    ) -> List[Dict[str, Any]]:
        """PostgreSQL 문서 쿼리

        Args:
            collection: 테이블 이름
            filters: 필터 목록 [(필드, 연산자, 값), ...]
            limit: 반환할 최대 문서 수
            order_by: 정렬 기준 필드
            order_direction: 정렬 방향 ('ASC' 또는 'DESC')

        Returns:
            문서 목록
        """
        try:
            # 세션 가져오기
            db = self.get_db()
            
            try:
                # 컬렉션에 따라 다른 모델 사용
                if collection == 'medical_literature':
                    model = MedicalLiterature
                elif collection == 'clinical_guidelines':
                    model = ClinicalGuideline
                elif collection == 'patient_records':
                    model = PatientRecord
                else:
                    raise ValueError(f'지원되지 않는 컬렉션: {collection}')
                
                # 쿼리 생성
                query = db.query(model)
                
                # 필터 적용
                if filters:
                    for field, op, value in filters:
                        column = getattr(model, field, None)
                        if not column:
                            logger.warning(f'필드 \'{field}\'가 모델 {model.__name__}에 존재하지 않습니다.')
                            continue
                        
                        if op == '==':
                            query = query.filter(column == value)
                        elif op == '>':
                            query = query.filter(column > value)
                        elif op == '<':
                            query = query.filter(column < value)
                        elif op == '>=':
                            query = query.filter(column >= value)
                        elif op == '<=':
                            query = query.filter(column <= value)
                        elif op == 'in':
                            query = query.filter(column.in_(value))
                        elif op == 'contains' and isinstance(value, str):
                            query = query.filter(column.contains(value))
                        else:
                            logger.warning(f'지원되지 않는 연산자: {op}')
                
                # 정렬 적용
                if order_by:
                    column = getattr(model, order_by, None)
                    if column:
                        if order_direction.upper() == 'ASC':
                            query = query.order_by(column.asc())
                        else:
                            query = query.order_by(column.desc())
                
                # 제한 적용
                if limit > 0:
                    query = query.limit(limit)
                
                # 쿼리 실행
                items = query.all()
                
                # 결과를 딕셔너리 리스트로 변환
                results = [self._model_to_dict(item) for item in items]
                
                logger.info(f'{collection} 테이블에서 {len(results)}개 문서 조회 완료')
                return results
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'{collection} 테이블 쿼리 중 오류 발생: {e}', exc_info=True)
            raise

    def _model_to_dict(self, model: Any) -> Dict[str, Any]:
        """SQLAlchemy 모델을 딕셔너리로 변환"""
        result = {}
        for column in model.__table__.columns:
            value = getattr(model, column.name)
            
            # datetime 객체를 문자열로 변환
            if isinstance(value, (time.struct_time, time.struct_time.__class__)):
                value = value.isoformat()
            
            result[column.name] = value
        
        # ID 추가
        result['id'] = model.id
        
        return result

    # --- 특정 컬렉션 작업을 위한 편의 메서드 ---

    async def add_patient_record(self, patient_id: str, data: Dict[str, Any]) -> str:
        """환자 기록 추가"""
        return await self.add_document('patient_records', data, doc_id=patient_id)

    async def get_patient_record(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """환자 기록 조회"""
        return await self.get_document('patient_records', patient_id)

    async def update_patient_record(self, patient_id: str, data: Dict[str, Any]) -> bool:
        """환자 기록 업데이트"""
        return await self.update_document('patient_records', patient_id, data)

    async def get_all_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """모든 환자 기록 가져오기"""
        return await self.query_documents('patient_records', limit=limit)

    async def add_medical_literature(self, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """의학 문헌 추가"""
        return await self.add_document('medical_literature', data, doc_id)

    async def get_medical_literature(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """의학 문헌 조회"""
        return await self.get_document('medical_literature', doc_id)

    async def search_medical_literature(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """의학 문헌 검색 (키워드 매칭)"""
        try:
            # LIKE 연산자를 사용하여 제목이나 초록에서 검색
            db = self.get_db()
            
            try:
                # SQL 쿼리 생성 (SQLAlchemy Core 사용)
                sql = text(
                    f"""
                    SELECT * FROM medical_literature
                    WHERE LOWER(title) LIKE LOWER(:query) OR LOWER(abstract) LIKE LOWER(:query)
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                )
                
                # 쿼리 실행
                result = db.execute(
                    sql, 
                    {'query': f'%{query}%', 'limit': limit}
                )
                
                # 결과를 딕셔너리 리스트로 변환
                docs = []
                for row in result:
                    doc = dict(row._mapping)
                    docs.append(doc)
                
                return docs
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'의학 문헌 검색 중 오류 발생: {e}', exc_info=True)
            return []

    async def add_clinical_guideline(self, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """임상 가이드라인 추가"""
        return await self.add_document('clinical_guidelines', data, doc_id)

    async def get_clinical_guideline(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """임상 가이드라인 조회"""
        return await self.get_document('clinical_guidelines', doc_id) 
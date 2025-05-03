"""
Firebase Firestore 데이터베이스 어댑터
"""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import uuid

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from db.firebase_config import COLLECTION_CLINICAL_GUIDELINES
from db.firebase_config import COLLECTION_MEDICAL_LITERATURE
from db.firebase_config import COLLECTION_PATIENT_RECORDS
from db.firebase_config import get_firestore_client

# 로깅 설정
logger = logging.getLogger(__name__)


class FirestoreDB:
    """Firebase Firestore를 사용하는 데이터베이스 어댑터"""

    def __init__(self):
        """Firestore 클라이언트 초기화"""
        self.db = get_firestore_client()
        logger.info('FirestoreDB 인스턴스 생성 완료')

    async def add_document(self, collection: str, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """Firestore에 문서 추가

        Args:
            collection: 컬렉션 이름
            data: 저장할 데이터
            doc_id: 사용할 문서 ID (없으면 자동 생성)

        Returns:
            생성된 문서 ID
        """
        try:
            # 타임스탬프 추가
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP

            # 문서 ID 생성 또는 사용
            if not doc_id:
                doc_id = str(uuid.uuid4())

            # 문서 추가
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.set(data)

            logger.info(f'문서가 {collection}/{doc_id}에 추가되었습니다.')
            return doc_id

        except Exception as e:
            logger.error(f'{collection} 컬렉션에 문서 추가 중 오류 발생: {e}', exc_info=True)
            raise

    async def get_document(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Firestore에서 특정 문서 가져오기

        Args:
            collection: 컬렉션 이름
            doc_id: 문서 ID

        Returns:
            문서 데이터 또는 None (존재하지 않는 경우)
        """
        try:
            doc_ref = self.db.collection(collection).document(doc_id)
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f'{collection}/{doc_id} 문서가 존재하지 않습니다.')
                return None

        except Exception as e:
            logger.error(f'{collection}/{doc_id} 문서 조회 중 오류 발생: {e}', exc_info=True)
            raise

    async def update_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> bool:
        """Firestore 문서 업데이트

        Args:
            collection: 컬렉션 이름
            doc_id: 문서 ID
            data: 업데이트할 데이터

        Returns:
            성공 여부
        """
        try:
            # 타임스탬프 업데이트
            data['updated_at'] = firestore.SERVER_TIMESTAMP

            # 문서 업데이트
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.update(data)

            logger.info(f'{collection}/{doc_id} 문서가 업데이트되었습니다.')
            return True

        except Exception as e:
            logger.error(f'{collection}/{doc_id} 문서 업데이트 중 오류 발생: {e}', exc_info=True)
            return False

    async def delete_document(self, collection: str, doc_id: str) -> bool:
        """Firestore 문서 삭제

        Args:
            collection: 컬렉션 이름
            doc_id: 문서 ID

        Returns:
            성공 여부
        """
        try:
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.delete()

            logger.info(f'{collection}/{doc_id} 문서가 삭제되었습니다.')
            return True

        except Exception as e:
            logger.error(f'{collection}/{doc_id} 문서 삭제 중 오류 발생: {e}', exc_info=True)
            return False

    async def query_documents(
        self, collection: str, filters: Optional[List[Tuple[str, str, Any]]] = None, limit: int = 10, order_by: Optional[str] = None, order_direction: str = 'DESCENDING'
    ) -> List[Dict[str, Any]]:
        """Firestore 문서 쿼리

        Args:
            collection: 컬렉션 이름
            filters: 필터 목록 [(필드, 연산자, 값), ...] - 연산자: ==, >, <, >=, <=, in, not-in, array-contains, array-contains-any
            limit: 반환할 최대 문서 수
            order_by: 정렬 기준 필드
            order_direction: 정렬 방향 ('ASCENDING' 또는 'DESCENDING')

        Returns:
            문서 목록
        """
        try:
            # 쿼리 생성
            query = self.db.collection(collection)

            # 필터 적용
            if filters:
                for field, op, value in filters:
                    query = query.where(filter=FieldFilter(field, op, value))

            # 정렬 적용
            if order_by:
                if order_direction == 'ASCENDING':
                    query = query.order_by(order_by, direction=firestore.Query.ASCENDING)
                else:
                    query = query.order_by(order_by, direction=firestore.Query.DESCENDING)

            # 제한 적용
            if limit > 0:
                query = query.limit(limit)

            # 쿼리 실행
            docs = query.stream()
            results = []

            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id  # 문서 ID 포함
                results.append(data)

            logger.info(f'{collection} 컬렉션에서 {len(results)}개 문서 조회 완료')
            return results

        except Exception as e:
            logger.error(f'{collection} 컬렉션 쿼리 중 오류 발생: {e}', exc_info=True)
            raise

    # --- 특정 컬렉션 작업을 위한 편의 메서드 ---

    async def add_patient_record(self, patient_id: str, data: Dict[str, Any]) -> str:
        """환자 기록 추가"""
        return await self.add_document(COLLECTION_PATIENT_RECORDS, data, doc_id=patient_id)

    async def get_patient_record(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """환자 기록 조회"""
        return await self.get_document(COLLECTION_PATIENT_RECORDS, patient_id)

    async def update_patient_record(self, patient_id: str, data: Dict[str, Any]) -> bool:
        """환자 기록 업데이트"""
        return await self.update_document(COLLECTION_PATIENT_RECORDS, patient_id, data)

    async def get_all_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """모든 환자 기록 가져오기"""
        return await self.query_documents(COLLECTION_PATIENT_RECORDS, limit=limit)

    async def add_medical_literature(self, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """의학 문헌 추가"""
        return await self.add_document(COLLECTION_MEDICAL_LITERATURE, data, doc_id)

    async def get_medical_literature(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """의학 문헌 조회"""
        return await self.get_document(COLLECTION_MEDICAL_LITERATURE, doc_id)

    async def search_medical_literature(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """의학 문헌 검색 (간단한 키워드 매칭)"""
        # 실제 구현에서는 Firestore 복합 쿼리 또는 확장 솔루션 사용 고려
        results = []
        
        try:
            # 제목, 초록, 본문에서 검색
            title_matches = await self.query_documents(
                COLLECTION_MEDICAL_LITERATURE,
                filters=[('title', '>=', query)],
                limit=limit
            )
            
            # 검색어가 포함된 문서만 필터링 (Firestore는 부분 문자열 검색을 직접 지원하지 않음)
            filtered_results = []
            query_lower = query.lower()
            
            for doc in title_matches:
                if query_lower in doc.get('title', '').lower() or query_lower in doc.get('abstract', '').lower():
                    # 이미 추가된 문서 ID인지 확인
                    if not any(r.get('id') == doc.get('id') for r in filtered_results):
                        filtered_results.append(doc)
            
            return filtered_results[:limit]
        
        except Exception as e:
            logger.error(f'의학 문헌 검색 중 오류 발생: {e}', exc_info=True)
            return []

    async def add_clinical_guideline(self, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """임상 가이드라인 추가"""
        return await self.add_document(COLLECTION_CLINICAL_GUIDELINES, data, doc_id)

    async def get_clinical_guideline(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """임상 가이드라인 조회"""
        return await self.get_document(COLLECTION_CLINICAL_GUIDELINES, doc_id) 
"""
Firebase Firestore를 사용하는 문서 및 환자 데이터 검색 시스템
"""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from db.firebase_db import FirestoreDB
from rag.mock_retriever import Document

# 로깅 설정
logger = logging.getLogger(__name__)


class FirebaseRetriever:
    """Firebase Firestore 기반 문서 검색 시스템"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Firebase Retriever 초기화"""
        self.db = FirestoreDB()
        logger.info('FirebaseRetriever 초기화 완료')

    async def retrieve_documents(self, query: str, limit: int = 3) -> List[Document]:
        """쿼리와 관련된 문서 검색

        Args:
            query: 검색어
            limit: 반환할 최대 문서 수

        Returns:
            관련 문서 목록
        """
        logger.info(f'문서 검색: \'{query}\'')
        
        try:
            # 의학 문헌 검색
            literature_results = await self.db.search_medical_literature(query, limit=limit)
            
            # 임상 가이드라인 검색 (간단한 구현)
            clinical_results = await self.db.query_documents(
                'clinical_guidelines',
                filters=[],  # 실제로는 더 복잡한 필터링 적용 필요
                limit=limit
            )
            
            # Document 객체로 변환
            documents = []
            
            # 의학 문헌 변환
            for doc in literature_results:
                documents.append(
                    Document(
                        id=doc.get('id', ''),
                        title=doc.get('title', '제목 없음'),
                        content=doc.get('abstract', '') or doc.get('full_text', ''),
                        metadata=doc.get('metadata', {}) or {
                            'authors': doc.get('authors', []),
                            'journal': doc.get('journal', ''),
                            'publication_date': doc.get('publication_date', ''),
                            'doi': doc.get('doi', ''),
                        },
                        score=1.0  # 실제 구현에서는 적절한 관련성 점수 계산 필요
                    )
                )
            
            # 임상 가이드라인 변환
            for doc in clinical_results:
                # 제목이나 내용에 검색어가 포함되어 있는지 확인
                if query.lower() in doc.get('title', '').lower() or query.lower() in doc.get('content', '').lower():
                    documents.append(
                        Document(
                            id=doc.get('id', ''),
                            title=doc.get('title', '제목 없음'),
                            content=doc.get('content', ''),
                            metadata=doc.get('metadata', {}) or {
                                'organization': doc.get('organization', ''),
                                'publish_date': doc.get('publish_date', ''),
                                'update_date': doc.get('update_date', ''),
                                'specialty': doc.get('specialty', ''),
                                'recommendation_level': doc.get('recommendation_level', ''),
                            },
                            score=0.9  # 가이드라인은 약간 낮은 점수 부여
                        )
                    )
            
            # 점수로 정렬하고 상위 문서 반환
            sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
            return sorted_docs[:limit]
            
        except Exception as e:
            logger.error(f'문서 검색 중 오류 발생: {e}', exc_info=True)
            return []
    
    async def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """환자 ID로 환자 데이터 검색

        Args:
            patient_id: 환자 ID

        Returns:
            환자 데이터 또는 None (존재하지 않는 경우)
        """
        logger.info(f'환자 데이터 검색: {patient_id}')
        
        try:
            return await self.db.get_patient_record(patient_id)
        except Exception as e:
            logger.error(f'환자 데이터 검색 중 오류 발생: {e}', exc_info=True)
            return None
    
    async def add_document(self, document: Document) -> str:
        """새 문서 추가

        Args:
            document: 추가할 문서

        Returns:
            생성된 문서 ID
        """
        try:
            # Document 객체를 딕셔너리로 변환
            doc_data = {
                'title': document.title,
                'content': document.content,
                'metadata': document.metadata,
            }
            
            # 문서 유형에 따라 다른 컬렉션에 저장
            # 실제 구현에서는 문서 유형을 더 세밀하게 구분해야 할 수 있음
            if 'journal' in document.metadata:
                # 의학 문헌으로 저장
                return await self.db.add_medical_literature(doc_data, document.id if document.id else None)
            else:
                # 임상 가이드라인 또는 일반 문서로 저장
                return await self.db.add_clinical_guideline(doc_data, document.id if document.id else None)
                
        except Exception as e:
            logger.error(f'문서 추가 중 오류 발생: {e}', exc_info=True)
            raise
    
    async def add_patient_data(self, patient_id: str, data: Dict[str, Any]) -> str:
        """환자 데이터 추가

        Args:
            patient_id: 환자 ID
            data: 환자 데이터

        Returns:
            생성된 문서 ID (환자 ID와 동일)
        """
        try:
            return await self.db.add_patient_record(patient_id, data)
        except Exception as e:
            logger.error(f'환자 데이터 추가 중 오류 발생: {e}', exc_info=True)
            raise
    
    async def get_all_patients(self, limit: int = 100) -> Dict[str, Dict[str, Any]]:
        """모든 환자 데이터 가져오기

        Returns:
            환자 ID를 키로 하고 환자 데이터를 값으로 하는 딕셔너리
        """
        try:
            patients_list = await self.db.get_all_patients(limit=limit)
            
            # 리스트를 딕셔너리로 변환
            patients_dict = {}
            for patient in patients_list:
                patient_id = patient.get('id', '')
                if patient_id:
                    # ID 필드 제거 후 저장
                    patient_data = {k: v for k, v in patient.items() if k != 'id'}
                    patients_dict[patient_id] = patient_data
            
            return patients_dict
            
        except Exception as e:
            logger.error(f'모든 환자 데이터 조회 중 오류 발생: {e}', exc_info=True)
            return {} 
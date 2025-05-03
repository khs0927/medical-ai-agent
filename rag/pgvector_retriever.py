from __future__ import annotations

'''
pgvector 기반 검색 모듈 (Phase 2-3)

실제 PostgreSQL + pgvector 벡터 DB 연결을 위한 구현.
현재는 구현 예시만 포함하고 있으며, .env에 DB 정보 추가 후 활성화 가능.
'''

import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import asyncpg
# sentence-transformers 임베딩 모델 사용
from sentence_transformers import SentenceTransformer

from rag.mock_retriever import Document

logger = logging.getLogger(__name__)


class PGVectorRetriever:
    '''PostgreSQL + pgvector 벡터 DB 기반 검색 구현'''

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        '''DB 설정 로드 및 임베딩 모델 초기화'''
        self.config = config or {}
        self.pool = None
        
        # DB 접속 정보
        self.db_config = {
            'host': os.getenv('PGVECTOR_HOST', self.config.get('host', 'localhost')),
            'port': os.getenv('PGVECTOR_PORT', self.config.get('port', 5432)),
            'database': os.getenv('PGVECTOR_DB', self.config.get('database', 'medigenius')),
            'user': os.getenv('PGVECTOR_USER', self.config.get('user', 'postgres')),
            'password': os.getenv('PGVECTOR_PASSWORD', self.config.get('password', 'postgres')),
        }
        
        # 임베딩 모델 로드
        self.embedding_model = self._load_embedding_model()
        
        logger.info(f'PGVectorRetriever 초기화: {self.db_config[\'host\']}:{self.db_config[\'port\']}/{self.db_config[\'database\']}')

    def _load_embedding_model(self) -> SentenceTransformer:
        '''임베딩 모델 로드 - 의학 도메인 특화 모델 사용'''
        # model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'  # 의학 도메인 특화
        model_name = 'all-MiniLM-L6-v2'  # 더 작은 모델 (테스트용)
        return SentenceTransformer(model_name)

    async def _get_db_pool(self) -> asyncpg.Pool:
        '''DB 풀 생성 또는 재사용'''
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(**self.db_config)
                # pgvector 확장 활성화 확인
                async with self.pool.acquire() as conn:
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
                logger.info('벡터 DB 연결 성공')
            except Exception as e:
                logger.error(f'벡터 DB 연결 실패: {e}')
                # 실패해도 작동할 수 있도록 폴백 구현 가능
                raise
        return self.pool

    def _create_embedding(self, text: str) -> List[float]:
        '''텍스트를 벡터로 변환'''
        return self.embedding_model.encode(text).tolist()

    async def retrieve_documents(self, query: str, limit: int = 3) -> List[Document]:
        '''쿼리와 관련된 문서 검색 (벡터 유사도 기반)'''
        logger.info(f'벡터 DB 문서 검색: {query}')
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self._create_embedding(query)
            
            # DB 연결
            pool = await self._get_db_pool()
            
            # 문헌 및 가이드라인 검색
            async with pool.acquire() as conn:
                # 문헌 검색
                literature_results = await conn.fetch('''
                    SELECT id, title, abstract as content, 
                           authors, publication_date, journal, doi,
                           embedding <=> $1 as distance
                    FROM medical_literature
                    ORDER BY distance
                    LIMIT $2
                ''', query_embedding, limit)
                
                # 가이드라인 검색
                guideline_results = await conn.fetch('''
                    SELECT id, title, content, 
                           organization, specialty, recommendation_level,
                           embedding <=> $1 as distance
                    FROM clinical_guidelines
                    ORDER BY distance
                    LIMIT $2
                ''', query_embedding, limit)
            
            # 결과 병합 및 Document 객체로 변환
            documents = []
            
            for row in literature_results:
                # 거리를 유사도 점수로 변환 (거리가 작을수록 유사도 높음)
                similarity = 1.0 / (1.0 + float(row['distance']))
                
                documents.append(Document(
                    id=f'lit_{row[\'id\']}',
                    title=row['title'],
                    content=row['content'],
                    metadata={
                        'type': 'literature',
                        'authors': row['authors'],
                        'publication_date': row['publication_date'].isoformat() if row['publication_date'] else None,
                        'journal': row['journal'],
                        'doi': row['doi']
                    },
                    score=similarity
                ))
            
            for row in guideline_results:
                similarity = 1.0 / (1.0 + float(row['distance']))
                
                documents.append(Document(
                    id=f'guide_{row[\'id\']}',
                    title=row['title'],
                    content=row['content'],
                    metadata={
                        'type': 'guideline',
                        'organization': row['organization'],
                        'specialty': row['specialty'],
                        'recommendation_level': row['recommendation_level']
                    },
                    score=similarity
                ))
            
            # 유사도 점수로 정렬
            documents.sort(key=lambda doc: doc.score, reverse=True)
            
            return documents[:limit]
            
        except Exception as e:
            logger.error(f'벡터 검색 중 오류: {e}', exc_info=True)
            # 실패 시 빈 리스트 반환 또는 예외 처리 필요
            return []

    async def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        '''환자 ID로 환자 데이터 검색'''
        logger.info(f'환자 데이터 검색: {patient_id}')
        
        try:
            pool = await self._get_db_pool()
            
            async with pool.acquire() as conn:
                result = await conn.fetchrow('''
                    SELECT patient_id, demographics, medical_history, 
                           medications, lab_results, vitals
                    FROM patient_records
                    WHERE patient_id = $1
                ''', patient_id)
            
            if not result:
                return None
                
            return {
                'patient_id': result['patient_id'],
                'demographics': result['demographics'],
                'medical_history': result['medical_history'],
                'medications': result['medications'],
                'lab_results': result['lab_results'],
                'vitals': result['vitals']
            }
            
        except Exception as e:
            logger.error(f'환자 데이터 조회 중 오류: {e}', exc_info=True)
            return None

    async def add_document(self, document: Document) -> str:
        '''문서를 벡터 DB에 추가'''
        logger.info(f'문서 추가: {document.title}')
        
        try:
            # 임베딩 생성
            text_for_embedding = f'{document.title}. {document.content}'
            embedding = self._create_embedding(text_for_embedding)
            
            pool = await self._get_db_pool()
            
            # 문서 타입에 따라 저장 테이블 결정
            doc_type = document.metadata.get('type', 'literature')
            
            async with pool.acquire() as conn:
                if doc_type == 'guideline':
                    # 가이드라인 저장
                    result = await conn.fetchrow('''
                        INSERT INTO clinical_guidelines (
                            title, content, organization, specialty, 
                            recommendation_level, embedding
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id
                    ''', 
                    document.title,
                    document.content,
                    document.metadata.get('organization', ''),
                    document.metadata.get('specialty', ''),
                    document.metadata.get('recommendation_level', ''),
                    embedding
                    )
                    
                    doc_id = f'guide_{result[\'id\']}'
                    
                else:  # 기본 literature
                    # 문헌 저장
                    result = await conn.fetchrow('''
                        INSERT INTO medical_literature (
                            title, abstract, authors, journal, doi, embedding
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id
                    ''', 
                    document.title,
                    document.content,
                    document.metadata.get('authors', []),
                    document.metadata.get('journal', ''),
                    document.metadata.get('doi', ''),
                    embedding
                    )
                    
                    doc_id = f'lit_{result[\'id\']}'
            
            return doc_id
            
        except Exception as e:
            logger.error(f'문서 추가 중 오류: {e}', exc_info=True)
            # 실패 시 원본 ID 반환
            return document.id

    async def add_patient_data(self, patient_id: str, data: Dict[str, Any]) -> str:
        '''환자 데이터를 벡터 DB에 추가 또는 업데이트'''
        logger.info(f'환자 데이터 추가/업데이트: {patient_id}')
        
        try:
            # 환자 정보 텍스트 구성 및 임베딩 생성
            embedding_text = f'환자 ID: {patient_id}. '
            
            if 'demographics' in data:
                demo = data['demographics']
                embedding_text += f'나이: {demo.get(\'age\')}. 성별: {demo.get(\'gender\')}. '
                
            if 'medical_history' in data:
                history = data.get('medical_history', [])
                if isinstance(history, list):
                    conditions = []
                    for item in history:
                        if isinstance(item, dict) and 'condition' in item:
                            conditions.append(item['condition'])
                        elif isinstance(item, str):
                            conditions.append(item)
                    if conditions:
                        embedding_text += f'병력: {\', \'.join(conditions)}. '
                        
            if 'medications' in data:
                meds = data.get('medications', [])
                if isinstance(meds, list):
                    med_names = []
                    for med in meds:
                        if isinstance(med, dict) and 'name' in med:
                            med_names.append(med['name'])
                        elif isinstance(med, str):
                            med_names.append(med)
                    if med_names:
                        embedding_text += f'약물: {\', \'.join(med_names)}. '
            
            # 임베딩 생성
            embedding = self._create_embedding(embedding_text)
            
            # DB 업데이트/삽입
            pool = await self._get_db_pool()
            
            async with pool.acquire() as conn:
                # 먼저 기존 환자 검색
                existing = await conn.fetchrow('''
                    SELECT id FROM patient_records WHERE patient_id = $1
                ''', patient_id)
                
                if existing:
                    # 환자 레코드 업데이트
                    await conn.execute('''
                        UPDATE patient_records SET
                            demographics = $1,
                            medical_history = $2,
                            medications = $3,
                            lab_results = $4,
                            vitals = $5,
                            embedding = $6,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE patient_id = $7
                    ''',
                    data.get('demographics', {}),
                    data.get('medical_history', []),
                    data.get('medications', []),
                    data.get('lab_results', []),
                    data.get('vitals', {}),
                    embedding,
                    patient_id
                    )
                else:
                    # 새 환자 레코드 삽입
                    await conn.execute('''
                        INSERT INTO patient_records (
                            patient_id, demographics, medical_history,
                            medications, lab_results, vitals, embedding
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ''',
                    patient_id,
                    data.get('demographics', {}),
                    data.get('medical_history', []),
                    data.get('medications', []),
                    data.get('lab_results', []),
                    data.get('vitals', {}),
                    embedding
                    )
            
            return patient_id
            
        except Exception as e:
            logger.error(f'환자 데이터 추가 중 오류: {e}', exc_info=True)
            return patient_id 
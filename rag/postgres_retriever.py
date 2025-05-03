"""
PostgreSQL 기반 검색 시스템

PostgreSQL 데이터베이스를 사용한 텍스트 검색 및 임베딩 기반 유사도 검색 구현
"""
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import uuid

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import torch

from db.postgres_db import PostgresDB
from rag.base_retriever import BaseRetriever

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgresRetriever(BaseRetriever):
    """PostgreSQL 기반 문서 검색 시스템"""

    def __init__(self, collection_name: str = 'medical_literature'):
        """PostgreSQL 검색 시스템 초기화

        Args:
            collection_name: 검색할 컬렉션 이름
        """
        self.collection_name = collection_name
        self.db = PostgresDB()
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.embedding_device = 'cpu'
        
        # 임베딩 모델 초기화 시도
        self._initialize_embedding_model()
        
        logger.info(f'PostgreSQL 검색 시스템 초기화 완료: 컬렉션 \'{collection_name}\'')

    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            import torch
            from transformers import AutoModel
            from transformers import AutoTokenizer

            # 임베딩 모델 설정
            embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
            self.embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 의학 도메인에 특화된 임베딩 모델 목록
            medical_embedding_models = [
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',  # 의학 논문에 특화
                'allenai/scibert_scivocab_uncased',  # 과학 문헌 특화
                'sentence-transformers/all-MiniLM-L6-v2',  # 일반 목적 (가벼운 모델)
                'sentence-transformers/all-mpnet-base-v2',  # 일반 목적 (고성능)
            ]
            
            # 환경 변수로 지정된 모델이 없는 경우 특화 모델 시도
            if not embedding_model_name or embedding_model_name == 'auto':
                for model_name in medical_embedding_models:
                    try:
                        logger.info(f'임베딩 모델 시도: {model_name}')
                        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.embedding_model = AutoModel.from_pretrained(model_name)
                        embedding_model_name = model_name
                        break
                    except Exception as e:
                        logger.warning(f'임베딩 모델 로드 실패: {model_name} - {e}')
                        continue
            else:
                logger.info(f'지정된 임베딩 모델 로드: {embedding_model_name}')
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
                self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
            
            # 모델 로드 성공 확인
            if self.embedding_model and self.embedding_tokenizer:
                # GPU 사용 가능 시 모델 이동
                if self.embedding_device == 'cuda':
                    self.embedding_model = self.embedding_model.to('cuda')
                
                # 모델을 평가 모드로 설정 (드롭아웃 등 학습 기능 비활성화)
                self.embedding_model.eval()
                
                logger.info(f'임베딩 모델 초기화 완료: {embedding_model_name} (장치: {self.embedding_device})')
                return True
            else:
                logger.warning('임베딩 모델을 초기화하지 못했습니다. 기본 키워드 검색만 사용됩니다.')
                return False
                
        except ImportError as e:
            logger.warning(f'임베딩 모델 초기화 실패 (필요한 라이브러리 없음): {e}')
            return False
        except Exception as e:
            logger.error(f'임베딩 모델 초기화 중 오류 발생: {e}', exc_info=True)
            return False

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """텍스트에서 임베딩 생성

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 또는 None (오류 시)
        """
        if not self.embedding_model or not self.embedding_tokenizer:
            logger.warning('임베딩 모델이 초기화되지 않았습니다.')
            return None
            
        try:
            # 입력 텍스트 정리
            if not text or len(text.strip()) == 0:
                logger.warning('임베딩을 위한 빈 텍스트가 제공되었습니다.')
                return None
                
            # 토큰화 및 인코딩
            encoded_input = self.embedding_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # GPU 사용 가능 시 텐서 이동
            if self.embedding_device == 'cuda':
                encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}
            
            # 모델 추론 (그라디언트 계산 없이)
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)
                
            # 문장 임베딩 계산 (마지막 레이어의 첫 번째 토큰 [CLS] 사용)
            embedding = model_output.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # 정규화 (코사인 유사도 계산을 위해)
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            return normalized_embedding.tolist()
        
        except Exception as e:
            logger.error(f'임베딩 생성 중 오류 발생: {e}', exc_info=True)
            return None

    async def _save_embedding(self, doc_id: str, embedding_vector: List[float], doc_type: str) -> Optional[str]:
        """임베딩 벡터 저장

        Args:
            doc_id: 문서 ID
            embedding_vector: 임베딩 벡터
            doc_type: 문서 유형 (medical_literature, clinical_guidelines 등)

        Returns:
            임베딩 ID 또는 None (오류 시)
        """
        try:
            # 세션 가져오기
            db = self.db.get_db()
            
            try:
                # 임베딩 ID 생성
                embedding_id = f'emb_{str(uuid.uuid4())}'
                
                # 임베딩 모델 이름 가져오기
                model_name = 'unknown'
                if self.embedding_model and hasattr(self.embedding_model, 'name_or_path'):
                    model_name = self.embedding_model.name_or_path
                elif self.embedding_model and hasattr(self.embedding_model, 'config') and hasattr(self.embedding_model.config, 'name_or_path'):
                    model_name = self.embedding_model.config.name_or_path
                
                # SQL 쿼리 생성
                sql = text(
                    """
                    INSERT INTO embeddings (id, vector, model_name, document_type)
                    VALUES (:id, :vector, :model_name, :document_type)
                    RETURNING id
                    """
                )
                
                # 쿼리 실행
                result = db.execute(
                    sql,
                    {
                        'id': embedding_id,
                        'vector': embedding_vector,
                        'model_name': model_name,
                        'document_type': doc_type
                    }
                )
                
                # 결과 확인
                db.commit()
                
                # 문서 업데이트 (임베딩 ID 링크)
                update_sql = text(
                    f"""
                    UPDATE {doc_type}
                    SET embedding_id = :embedding_id
                    WHERE id = :doc_id
                    """
                )
                
                db.execute(update_sql, {'embedding_id': embedding_id, 'doc_id': doc_id})
                db.commit()
                
                logger.info(f'임베딩 저장 완료: {embedding_id} (문서: {doc_id})')
                return embedding_id
            
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f'임베딩 저장 중 SQL 오류: {e}')
                return None
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'임베딩 저장 중 오류 발생: {e}', exc_info=True)
            return None

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """의미론적 검색 수행 (임베딩 기반 유사도 검색)

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 목록
        """
        logger.info(f'의미론적 검색: \'{query}\', 컬렉션: {self.collection_name}, top_k: {top_k}')
        
        # 임베딩 모델 사용 가능 여부 확인
        if not self.embedding_model or not self.embedding_tokenizer:
            logger.warning('임베딩 모델이 사용 불가능합니다. 일반 검색으로 대체합니다.')
            return await self.search(query, top_k)
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self._generate_embedding(query)
            
            if not query_embedding:
                logger.warning('쿼리 임베딩 생성 실패. 일반 검색으로 대체합니다.')
                return await self.search(query, top_k)
            
            # 세션 가져오기
            db = self.db.get_db()
            
            try:
                # 컬렉션 매핑
                collection_mapping = {
                    'medical_literature': 'medical_literature',
                    'clinical_guidelines': 'clinical_guidelines'
                }
                
                # 매핑 확인
                if self.collection_name not in collection_mapping:
                    logger.warning(f'지원되지 않는 컬렉션: {self.collection_name}. 일반 검색으로 대체합니다.')
                    return await self.search(query, top_k)
                
                # 코사인 유사도 쿼리 실행
                # PostgreSQL 배열 내적 연산자(*)를 사용한 코사인 유사도 계산
                table_name = collection_mapping[self.collection_name]
                
                sql = text(
                    f"""
                    WITH similarity AS (
                        SELECT 
                            t.id, 
                            t.title,
                            CASE 
                                WHEN '{table_name}' = 'medical_literature' THEN t.abstract
                                WHEN '{table_name}' = 'clinical_guidelines' THEN t.content
                                ELSE NULL
                            END as content,
                            CASE
                                WHEN '{table_name}' = 'medical_literature' THEN 
                                    json_build_object(
                                        'authors', t.authors,
                                        'publication_date', t.publication_date,
                                        'journal', t.journal,
                                        'doi', t.doi
                                    )
                                WHEN '{table_name}' = 'clinical_guidelines' THEN 
                                    json_build_object(
                                        'organization', t.organization,
                                        'publish_date', t.publish_date,
                                        'update_date', t.update_date,
                                        'specialty', t.specialty,
                                        'recommendation_level', t.recommendation_level
                                    )
                                ELSE NULL
                            END as metadata,
                            e.vector * :query_vector AS similarity_score
                        FROM 
                            {table_name} t
                        JOIN 
                            embeddings e ON t.embedding_id = e.id
                        WHERE 
                            e.document_type = '{table_name}'
                    )
                    SELECT 
                        id, 
                        title, 
                        content,
                        metadata,
                        similarity_score
                    FROM 
                        similarity
                    WHERE 
                        similarity_score > 0.5
                    ORDER BY 
                        similarity_score DESC
                    LIMIT :top_k
                    """
                )
                
                # 쿼리 실행
                result = db.execute(
                    sql, 
                    {
                        'query_vector': query_embedding,
                        'top_k': top_k
                    }
                )
                
                # 결과 처리
                documents = []
                for row in result:
                    doc = {
                        'id': row.id,
                        'title': row.title,
                        'content': row.content,
                        'metadata': row.metadata,
                        'score': float(row.similarity_score)
                    }
                    documents.append(doc)
                
                # 결과가 충분하지 않은 경우 일반 검색으로 보완
                if len(documents) < top_k:
                    logger.info(f'의미론적 검색 결과 부족 ({len(documents)}개). 키워드 검색으로 보완합니다.')
                    
                    # 기존 문서 ID 목록
                    existing_ids = {doc['id'] for doc in documents}
                    
                    # 일반 검색 결과 가져오기
                    keyword_results = await self.search(query, top_k=top_k)
                    
                    # 중복되지 않는 결과만 추가
                    for doc in keyword_results:
                        if doc['id'] not in existing_ids and len(documents) < top_k:
                            # 기본 점수 조정 (의미론적 검색 결과보다 낮게)
                            doc['score'] = min(doc['score'], 0.7)
                            documents.append(doc)
                            existing_ids.add(doc['id'])
                
                # 점수별 내림차순 정렬
                documents = sorted(documents, key=lambda x: x['score'], reverse=True)
                
                logger.info(f'의미론적 검색 완료: {len(documents)}개 결과')
                return documents
                
            except SQLAlchemyError as e:
                logger.error(f'의미론적 검색 중 SQL 오류: {e}', exc_info=True)
                
                # 대체 검색 사용
                logger.info('일반 검색으로 대체합니다.')
                return await self.search(query, top_k)
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f'의미론적 검색 중 오류 발생: {e}', exc_info=True)
            
            # 대체 검색 사용
            logger.info('일반 검색으로 대체합니다.')
            return await self.search(query, top_k)

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """키워드 기반 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 목록
        """
        logger.info(f'PostgreSQL 키워드 검색: \'{query}\', 컬렉션: {self.collection_name}, top_k: {top_k}')
        
        try:
            # 컬렉션 선택에 따라 다른 메서드 호출
            if self.collection_name == 'medical_literature':
                # 의학 문헌 검색
                results = await self.db.search_medical_literature(query, limit=top_k)
            elif self.collection_name == 'clinical_guidelines':
                # 임상 가이드라인 검색 (내용 포함 검색)
                keywords = query.split()
                filters = []
                
                # 각 키워드에 대해 컨텐츠 검색 필터 적용
                for keyword in keywords:
                    if len(keyword) >= 3:  # 너무 짧은 키워드는 제외
                        filters.append(('content', 'contains', keyword))
                
                # 필터가 없으면 전체 조회
                if not filters:
                    results = await self.db.query_documents(self.collection_name, limit=top_k)
                else:
                    # 필터를 적용한 검색
                    results = await self.db.query_documents(self.collection_name, filters=filters, limit=top_k)
            else:
                # 기타 컬렉션 기본 쿼리
                results = await self.db.query_documents(self.collection_name, limit=top_k)
            
            # 검색 결과 처리 및 반환
            processed_results = []
            for doc in results:
                # 공통 필드
                result = {
                    'id': doc.get('id', ''),
                    'score': 1.0,  # 기본 점수 (임베딩 없이는 정확한 점수 산출 어려움)
                }
                
                # 컬렉션별 다른 필드 처리
                if self.collection_name == 'medical_literature':
                    result.update({
                        'title': doc.get('title', ''),
                        'content': doc.get('abstract', '') or doc.get('full_text', ''),
                        'metadata': {
                            'authors': doc.get('authors', []),
                            'publication_date': doc.get('publication_date', ''),
                            'journal': doc.get('journal', ''),
                            'doi': doc.get('doi', '')
                        }
                    })
                elif self.collection_name == 'clinical_guidelines':
                    result.update({
                        'title': doc.get('title', ''),
                        'content': doc.get('content', ''),
                        'metadata': {
                            'organization': doc.get('organization', ''),
                            'publish_date': doc.get('publish_date', ''),
                            'update_date': doc.get('update_date', ''),
                            'specialty': doc.get('specialty', ''),
                            'recommendation_level': doc.get('recommendation_level', '')
                        }
                    })
                else:
                    # 기타 컬렉션 기본 처리
                    result.update({
                        'title': doc.get('title', ''),
                        'content': doc.get('content', '') or doc.get('text', '') or doc.get('abstract', ''),
                        'metadata': doc.get('metadata', {}) or {}
                    })
                
                processed_results.append(result)
            
            # 검색 결과 점수 산출 (간단한 키워드 매칭 기반)
            query_terms = set(query.lower().split())
            for result in processed_results:
                title = result.get('title', '').lower()
                content = result.get('content', '').lower()
                
                # 제목과 내용에서 일치하는 쿼리 용어 수 계산
                title_matches = sum(1 for term in query_terms if term in title)
                content_matches = sum(1 for term in query_terms if term in content)
                
                # 가중치 적용 (제목 일치에 더 높은 가중치)
                score = (title_matches * 2 + content_matches) / (len(query_terms) * 3)
                result['score'] = min(max(score, 0.5), 1.0)  # 점수 범위 제한 (0.5-1.0)
            
            # 점수별 내림차순 정렬
            processed_results = sorted(processed_results, key=lambda x: x['score'], reverse=True)
            
            logger.info(f'키워드 검색 완료: {len(processed_results)}개 결과')
            return processed_results
        
        except Exception as e:
            logger.error(f'PostgreSQL 키워드 검색 중 오류 발생: {e}', exc_info=True)
            return []

    async def add_document(self, document: Dict[str, Any]) -> str:
        """문서 추가 및 임베딩 생성

        Args:
            document: 추가할 문서 데이터

        Returns:
            문서 ID
        """
        try:
            # 컬렉션에 따라 다른 메서드 호출
            doc_id = None
            
            if self.collection_name == 'medical_literature':
                doc_id = await self.db.add_medical_literature(document)
            elif self.collection_name == 'clinical_guidelines':
                doc_id = await self.db.add_clinical_guideline(document)
            else:
                doc_id = await self.db.add_document(self.collection_name, document)
            
            # 문서 추가 성공 시 임베딩 생성 및 저장
            if doc_id and self.embedding_model and self.embedding_tokenizer:
                # 임베딩할 텍스트 준비
                text_to_embed = ''
                
                if self.collection_name == 'medical_literature':
                    # 제목 + 초록으로 임베딩
                    title = document.get('title', '')
                    abstract = document.get('abstract', '')
                    text_to_embed = f'{title}\n\n{abstract}'
                elif self.collection_name == 'clinical_guidelines':
                    # 제목 + 내용으로 임베딩
                    title = document.get('title', '')
                    content = document.get('content', '')
                    # 콘텐츠가 너무 길 경우 첫 1000자만 사용
                    if len(content) > 1000:
                        content = content[:1000]
                    text_to_embed = f'{title}\n\n{content}'
                else:
                    # 기타 컬렉션의 경우 제목 + 내용
                    title = document.get('title', '')
                    content = document.get('content', '') or document.get('text', '')
                    text_to_embed = f'{title}\n\n{content}'
                
                # 임베딩 생성
                if text_to_embed:
                    embedding_vector = self._generate_embedding(text_to_embed)
                    
                    if embedding_vector:
                        # 임베딩 저장
                        await self._save_embedding(
                            doc_id=doc_id,
                            embedding_vector=embedding_vector,
                            doc_type=self.collection_name
                        )
            
            return doc_id
        
        except Exception as e:
            logger.error(f'문서 추가 중 오류 발생: {e}', exc_info=True)
            raise

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """문서 조회

        Args:
            doc_id: 문서 ID

        Returns:
            문서 데이터 또는 None (존재하지 않는 경우)
        """
        try:
            return await self.db.get_document(self.collection_name, doc_id)
        except Exception as e:
            logger.error(f'문서 조회 중 오류 발생: {e}', exc_info=True)
            return None

    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """문서 업데이트 및 임베딩 갱신

        Args:
            doc_id: 문서 ID
            document: 업데이트할 문서 데이터

        Returns:
            성공 여부
        """
        try:
            # 문서 업데이트
            result = await self.db.update_document(self.collection_name, doc_id, document)
            
            # 업데이트 성공 시 임베딩 갱신
            if result and self.embedding_model and self.embedding_tokenizer:
                # 임베딩할 텍스트 준비
                text_to_embed = ''
                
                # 업데이트된 전체 문서 조회
                updated_doc = await self.db.get_document(self.collection_name, doc_id)
                
                if updated_doc:
                    if self.collection_name == 'medical_literature':
                        # 제목 + 초록으로 임베딩
                        title = updated_doc.get('title', '')
                        abstract = updated_doc.get('abstract', '')
                        text_to_embed = f'{title}\n\n{abstract}'
                    elif self.collection_name == 'clinical_guidelines':
                        # 제목 + 내용으로 임베딩
                        title = updated_doc.get('title', '')
                        content = updated_doc.get('content', '')
                        # 콘텐츠가 너무 길 경우 첫 1000자만 사용
                        if len(content) > 1000:
                            content = content[:1000]
                        text_to_embed = f'{title}\n\n{content}'
                    else:
                        # 기타 컬렉션의 경우 제목 + 내용
                        title = updated_doc.get('title', '')
                        content = updated_doc.get('content', '') or updated_doc.get('text', '')
                        text_to_embed = f'{title}\n\n{content}'
                
                # 임베딩 생성
                if text_to_embed:
                    embedding_vector = self._generate_embedding(text_to_embed)
                    
                    if embedding_vector:
                        # 기존 임베딩 ID 확인
                        existing_embedding_id = updated_doc.get('embedding_id') if updated_doc else None
                        
                        if existing_embedding_id:
                            # 기존 임베딩 업데이트
                            db = self.db.get_db()
                            try:
                                update_sql = text(
                                    """
                                    UPDATE embeddings
                                    SET vector = :vector, updated_at = NOW()
                                    WHERE id = :id
                                    """
                                )
                                
                                db.execute(update_sql, {'id': existing_embedding_id, 'vector': embedding_vector})
                                db.commit()
                                logger.info(f'임베딩 업데이트 완료: {existing_embedding_id}')
                            except Exception as e:
                                db.rollback()
                                logger.error(f'임베딩 업데이트 실패: {e}')
                            finally:
                                db.close()
                        else:
                            # 새 임베딩 생성
                            await self._save_embedding(
                                doc_id=doc_id,
                                embedding_vector=embedding_vector,
                                doc_type=self.collection_name
                            )
            
            return result
        except Exception as e:
            logger.error(f'문서 업데이트 중 오류 발생: {e}', exc_info=True)
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """문서 삭제 및 관련 임베딩 삭제

        Args:
            doc_id: 문서 ID

        Returns:
            성공 여부
        """
        try:
            # 삭제 전 문서 조회하여 임베딩 ID 획득
            document = await self.db.get_document(self.collection_name, doc_id)
            embedding_id = document.get('embedding_id') if document else None
            
            # 문서 삭제
            result = await self.db.delete_document(self.collection_name, doc_id)
            
            # 연결된 임베딩 삭제
            if result and embedding_id:
                db = self.db.get_db()
                try:
                    delete_sql = text(
                        """
                        DELETE FROM embeddings
                        WHERE id = :id
                        """
                    )
                    
                    db.execute(delete_sql, {'id': embedding_id})
                    db.commit()
                    logger.info(f'임베딩 삭제 완료: {embedding_id}')
                except Exception as e:
                    db.rollback()
                    logger.error(f'임베딩 삭제 실패: {e}')
                finally:
                    db.close()
            
            return result
        except Exception as e:
            logger.error(f'문서 삭제 중 오류 발생: {e}', exc_info=True)
            return False 
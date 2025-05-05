'''
웹 검색 및 스크래핑 기반 RAG 검색기 구현
'''

import logging
import os
from typing import Any, Dict, List, Optional, Union
import uuid

from rag.mock_retriever import Document
from utils.web_scraper import MedicalWebScraper
from utils.web_search import MedicalWebSearch

# 로깅 설정
logger = logging.getLogger(__name__)


class WebRetriever:
    '''웹 검색 및 스크래핑을 통한 RAG 검색기'''

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        '''웹 RAG 검색기 초기화
        
        Args:
            config: 설정 (API 키 등)
        '''
        self.config = config or {}
        
        # 웹 검색 클라이언트 초기화
        google_api_key = self.config.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
        google_cse_id = self.config.get('google_cse_id') or os.getenv('GOOGLE_CSE_ID')
        
        self.web_search = MedicalWebSearch(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
        
        # 웹 스크래퍼 초기화
        self.web_scraper = MedicalWebScraper()
        
        # 캐싱 설정
        self.use_cache = self.config.get('use_cache', True)
        self.cache = {}
        
        logger.info('웹 RAG 검색기 초기화 완료')

    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Document]:
        '''쿼리와 관련된 문서를 웹에서 검색하여 반환
        
        Args:
            query: 사용자 쿼리
            limit: 최대 결과 수
            
        Returns:
            관련 문서 목록
        '''
        logger.info(f"웹 검색 실행: {query}")
        
        # 캐시 확인
        cache_key = f'web_retrieve_{query}_{limit}'
        if self.use_cache and cache_key in self.cache:
            logger.info(f'캐시에서 검색 결과 로드: {query}')
            return self.cache[cache_key]
        
        # 검색 결과와 가이드라인 모두 검색
        search_results = await self._search_combined(query, limit)
        
        # 캐시 저장
        if self.use_cache:
            self.cache[cache_key] = search_results
        
        return search_results

    async def _search_combined(self, query: str, limit: int) -> List[Document]:
        '''의학 정보와 가이드라인을 동시에 검색
        
        Args:
            query: 검색 쿼리
            limit: 최대 결과 수
            
        Returns:
            통합 검색 결과
        '''
        # 일반 의학 정보 검색
        medical_results = self.web_search.search_medical_info(query, limit // 2 + 1)
        
        # 임상 가이드라인 검색
        guideline_results = self.web_search.search_clinical_guidelines(query, limit // 2)
        
        # 의학 저널 검색
        journal_results = self.web_search.search_medical_journals(query, limit // 2)
        
        # 검색 결과 처리
        all_documents = []
        
        # 일반 의학 정보 처리
        for i, result in enumerate(medical_results):
            try:
                if i >= limit // 3:  # 균형을 위해 일정 비율만 사용
                    break
                    
                # 검색 결과 URL에서 콘텐츠 스크래핑
                doc_data = await self._scrape_and_process(result['link'], 'medical')
                if doc_data:
                    all_documents.append(doc_data)
            except Exception as e:
                logger.warning(f'의학 정보 처리 오류: {e}')
        
        # 가이드라인 처리
        for i, result in enumerate(guideline_results):
            try:
                if i >= limit // 3:  # 균형을 위해 일정 비율만 사용
                    break
                    
                # 가이드라인 URL에서 콘텐츠 스크래핑
                doc_data = await self._scrape_and_process(result['link'], 'guideline')
                if doc_data:
                    all_documents.append(doc_data)
            except Exception as e:
                logger.warning(f'가이드라인 처리 오류: {e}')
        
        # 저널 처리
        for i, result in enumerate(journal_results):
            try:
                if i >= limit // 3:  # 균형을 위해 일정 비율만 사용
                    break
                    
                # 저널 URL에서 콘텐츠 스크래핑
                doc_data = await self._scrape_and_process(result['link'], 'journal')
                if doc_data:
                    all_documents.append(doc_data)
            except Exception as e:
                logger.warning(f'저널 처리 오류: {e}')
        
        # 중복 제거 및 제한
        unique_documents = self._remove_duplicates(all_documents)
        limited_documents = unique_documents[:limit]
        
        logger.info(f'웹 검색 결과: {len(limited_documents)}개 문서 반환')
        return limited_documents

    async def _scrape_and_process(self, url: str, doc_type: str) -> Optional[Document]:
        '''URL에서 콘텐츠를 스크래핑하고 Document 객체로 변환
        
        Args:
            url: 스크래핑할 URL
            doc_type: 문서 유형 (medical, guideline, journal)
            
        Returns:
            처리된 Document 객체 또는 None (실패 시)
        '''
        try:
            # 캐시 확인
            cache_key = f'scrape_{url}'
            if self.use_cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            # 문서 유형에 따라 스크래핑 방법 결정
            if doc_type == 'journal':
                scraped_data = self.web_scraper.scrape_medical_journal(url)
            elif doc_type == 'guideline':
                scraped_data = self.web_scraper.scrape_medical_guidelines(url)
            else:
                scraped_data = self.web_scraper.scrape_article(url)
            
            # 스크래핑 실패 확인
            if 'error' in scraped_data:
                logger.warning(f"웹 검색 결과 스크랩 실패: {url}")
                return None
            
            # 타이틀과 콘텐츠 확인
            title = scraped_data.get('title', '')
            
            # 콘텐츠 추출 (문서 유형에 따라 다른 필드 사용)
            if doc_type == 'journal':
                content = scraped_data.get('abstract', '') or scraped_data.get('content', '')
                # 초록과 본문이 모두 있으면 둘 다 사용
                if scraped_data.get('abstract') and scraped_data.get('content'):
                    content = f"{scraped_data.get('abstract')}\n\n{scraped_data.get('content')}"
            else:
                content = scraped_data.get('content', '') or scraped_data.get('text', '')
            
            # 콘텐츠 없으면 실패
            if not content or not title:
                logger.warning(f'콘텐츠 없음: {url}')
                return None
            
            # 메타데이터 구성
            metadata = {
                'source_url': url,
                'type': doc_type,
                'date': scraped_data.get('publication_date', ''),
            }
            
            # 문서 유형별 추가 메타데이터
            if doc_type == 'journal':
                metadata.update({
                    'authors': scraped_data.get('authors', []),
                    'journal': scraped_data.get('journal', ''),
                    'doi': scraped_data.get('doi', ''),
                })
            elif doc_type == 'guideline':
                metadata.update({
                    'organization': scraped_data.get('organization', ''),
                    'recommendations': scraped_data.get('recommendations', []),
                })
            
            # Document 객체 생성
            document = Document(
                id=f'web_{doc_type}_{uuid.uuid4().hex[:8]}',
                title=title,
                content=content,
                metadata=metadata,
                score=1.0  # 기본 점수
            )
            
            # 캐시 저장
            if self.use_cache:
                self.cache[cache_key] = document
            
            return document
            
        except Exception as e:
            logger.error(f'스크래핑 오류: {url} - {e}', exc_info=True)
            return None

    def _remove_duplicates(self, documents: List[Document]) -> List[Document]:
        '''중복 문서 제거
        
        Args:
            documents: 문서 목록
            
        Returns:
            중복이 제거된 문서 목록
        '''
        unique_docs = {}
        
        for doc in documents:
            # 타이틀 기반 중복 체크
            title_key = doc.title.lower().strip()
            
            if title_key not in unique_docs:
                unique_docs[title_key] = doc
            else:
                # 기존 문서가 있으면 더 긴 콘텐츠나 더 높은 점수로 대체
                existing_doc = unique_docs[title_key]
                if (len(doc.content) > len(existing_doc.content) or 
                    doc.score > existing_doc.score):
                    unique_docs[title_key] = doc
        
        return list(unique_docs.values())

    async def _update_metadata(self, doc: Document) -> Document:
        '''문서 메타데이터 업데이트 (언어 감지, 키워드 추출 등)
        
        Args:
            doc: 업데이트할 문서
            
        Returns:
            업데이트된 문서
        '''
        # 이 메서드는 필요에 따라 향후 구현 예정
        return doc

    async def add_document(self, document: Document) -> str:
        '''문서 추가 (웹 기반 검색기에서는 지원되지 않음)'''
        logger.warning('WebRetriever는 add_document를 지원하지 않습니다')
        return document.id

    async def search_with_filter(self, query: str, filters: Dict[str, Any], limit: int = 5) -> List[Document]:
        '''필터링과 함께 웹 검색 수행
        
        Args:
            query: 검색 쿼리
            filters: 필터 조건
            limit: 최대 결과 수
            
        Returns:
            필터링된 검색 결과
        '''
        # 모든 결과 검색
        all_results = await self.retrieve_documents(query, limit=limit*2)  # 필터링 후에도 충분한 결과를 위해 2배로 검색
        
        # 필터 적용
        filtered_results = []
        for doc in all_results:
            match = True
            
            # 각 필터 조건 검사
            for key, value in filters.items():
                if key in doc.metadata:
                    # 리스트 필터
                    if isinstance(value, list):
                        if not any(v in doc.metadata[key] for v in value):
                            match = False
                            break
                    # 문자열 또는 기타 값 필터
                    elif doc.metadata[key] != value:
                        match = False
                        break
            
            if match:
                filtered_results.append(doc)
                
                # 충분한 결과를 찾으면 중단
                if len(filtered_results) >= limit:
                    break
        
        return filtered_results[:limit] 
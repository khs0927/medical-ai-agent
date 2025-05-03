#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WebRetriever 모듈 단위 테스트
"""

import unittest
import os
import sys
import logging
import asyncio
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 환경 변수 로드
load_dotenv()

# 테스트 대상 모듈 가져오기
from rag.web_retriever import WebRetriever
from rag.mock_retriever import Document

class TestWebRetriever(unittest.TestCase):
    """WebRetriever 클래스 단위 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.retriever = WebRetriever()
    
    @patch('rag.web_retriever.MedicalWebSearch')
    @patch('rag.web_retriever.MedicalWebScraper')
    def test_init(self, mock_scraper, mock_search):
        """초기화 테스트"""
        # Mock 설정
        mock_search_instance = MagicMock()
        mock_scraper_instance = MagicMock()
        mock_search.return_value = mock_search_instance
        mock_scraper.return_value = mock_scraper_instance
        
        # 웹 리트리버 초기화 테스트
        retriever = WebRetriever(
            api_key='test_key', 
            cse_id='test_id'
        )
        
        # 검증
        mock_search.assert_called_once_with(
            google_api_key='test_key',
            google_cse_id='test_id'
        )
        mock_scraper.assert_called_once()
        self.assertEqual(retriever.search, mock_search_instance)
        self.assertEqual(retriever.scraper, mock_scraper_instance)
    
    @patch('rag.web_retriever.MedicalWebSearch')
    @patch('rag.web_retriever.MedicalWebScraper')
    def test_init_env_variables(self, mock_scraper, mock_search):
        """환경 변수 기반 초기화 테스트"""
        # 환경 변수 설정
        original_api_key = os.environ.get('GOOGLE_API_KEY')
        original_cse_id = os.environ.get('GOOGLE_CSE_ID')
        os.environ['GOOGLE_API_KEY'] = 'env_api_key'
        os.environ['GOOGLE_CSE_ID'] = 'env_cse_id'
        
        try:
            # Mock 설정
            mock_search_instance = MagicMock()
            mock_scraper_instance = MagicMock()
            mock_search.return_value = mock_search_instance
            mock_scraper.return_value = mock_scraper_instance
            
            # 웹 리트리버 초기화 테스트 (API 키, CSE ID 지정 없이)
            retriever = WebRetriever()
            
            # 검증
            mock_search.assert_called_once_with(
                google_api_key='env_api_key',
                google_cse_id='env_cse_id'
            )
            
        finally:
            # 환경 변수 복원
            if original_api_key:
                os.environ['GOOGLE_API_KEY'] = original_api_key
            else:
                del os.environ['GOOGLE_API_KEY']
                
            if original_cse_id:
                os.environ['GOOGLE_CSE_ID'] = original_cse_id
            else:
                del os.environ['GOOGLE_CSE_ID']
    
    @patch('rag.web_retriever.WebRetriever.search_and_scrape')
    async def test_retrieve_documents(self, mock_search_scrape):
        """문서 검색 테스트"""
        # Mock 검색 결과 설정
        mock_results = [
            {
                'title': '검색 결과 1',
                'content': '검색 결과 내용 1',
                'url': 'https://example.com/1',
                'source': '소스 1'
            },
            {
                'title': '검색 결과 2',
                'content': '검색 결과 내용 2',
                'url': 'https://example.com/2',
                'source': '소스 2'
            }
        ]
        
        mock_search_scrape.return_value = mock_results
        
        # 문서 검색 테스트
        documents = await self.retriever.retrieve_documents('테스트 쿼리', limit=2)
        
        # 검증
        mock_search_scrape.assert_called_once_with('테스트 쿼리', search_type='medical', num_results=2)
        
        self.assertEqual(len(documents), 2)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].title, '검색 결과 1')
        self.assertEqual(documents[0].content, '검색 결과 내용 1')
        self.assertEqual(documents[0].metadata['source_url'], 'https://example.com/1')
        self.assertEqual(documents[0].metadata['source'], '소스 1')
        self.assertEqual(documents[0].metadata['type'], 'web')
    
    @patch('rag.web_retriever.WebRetriever.search_web')
    @patch('rag.web_retriever.WebRetriever.scrape_search_results')
    async def test_search_and_scrape(self, mock_scrape, mock_search):
        """검색 및 스크래핑 통합 테스트"""
        # Mock 응답 설정
        mock_search_results = [
            {
                'title': '검색 제목 1',
                'link': 'https://example.com/1',
                'snippet': '발췌 1'
            },
            {
                'title': '검색 제목 2',
                'link': 'https://example.com/2',
                'snippet': '발췌 2'
            }
        ]
        
        mock_scrape_results = [
            {
                'title': '스크랩 제목 1',
                'content': '스크랩 내용 1',
                'url': 'https://example.com/1',
                'source': '소스 1'
            },
            {
                'title': '스크랩 제목 2',
                'content': '스크랩 내용 2',
                'url': 'https://example.com/2',
                'source': '소스 2'
            }
        ]
        
        mock_search.return_value = mock_search_results
        mock_scrape.return_value = mock_scrape_results
        
        # 검색 및 스크래핑 테스트
        results = await self.retriever.search_and_scrape('테스트 쿼리', search_type='medical', num_results=2)
        
        # 검증
        mock_search.assert_called_once_with('테스트 쿼리', search_type='medical', num_results=2)
        mock_scrape.assert_called_once_with(mock_search_results)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], '스크랩 제목 1')
        self.assertEqual(results[0]['content'], '스크랩 내용 1')
        self.assertEqual(results[0]['url'], 'https://example.com/1')
    
    @patch('utils.web_search.MedicalWebSearch.search')
    @patch('utils.web_search.MedicalWebSearch.search_medical_info')
    @patch('utils.web_search.MedicalWebSearch.search_medical_journals')
    async def test_search_web(self, mock_journals, mock_medical, mock_general):
        """웹 검색 메서드 테스트"""
        # Mock 응답 설정
        mock_general.return_value = [{'title': '일반 검색 결과'}]
        mock_medical.return_value = [{'title': '의학 검색 결과'}]
        mock_journals.return_value = [{'title': '저널 검색 결과'}]
        
        self.retriever.search = MagicMock()
        self.retriever.search.search = mock_general
        self.retriever.search.search_medical_info = mock_medical
        self.retriever.search.search_medical_journals = mock_journals
        
        # 여러 검색 유형 테스트
        general_results = await self.retriever.search_web('테스트 쿼리', search_type='general')
        medical_results = await self.retriever.search_web('테스트 쿼리', search_type='medical')
        journal_results = await self.retriever.search_web('테스트 쿼리', search_type='journal')
        
        # 검증
        mock_general.assert_called_once()
        mock_medical.assert_called_once()
        mock_journals.assert_called_once()
        
        self.assertEqual(general_results[0]['title'], '일반 검색 결과')
        self.assertEqual(medical_results[0]['title'], '의학 검색 결과')
        self.assertEqual(journal_results[0]['title'], '저널 검색 결과')
    
    @patch('utils.web_scraper.MedicalWebScraper.scrape_medical_journal')
    async def test_scrape_search_results(self, mock_scrape):
        """검색 결과 스크래핑 테스트"""
        # Mock 응답 설정
        mock_scrape.side_effect = [
            {
                'title': '스크랩 제목 1',
                'content': '스크랩 내용 1',
                'abstract': '초록 1',
                'source': '소스 1'
            },
            {
                'error': '스크래핑 실패'
            }
        ]
        
        self.retriever.scraper = MagicMock()
        self.retriever.scraper.scrape_medical_journal = mock_scrape
        
        # 검색 결과
        search_results = [
            {
                'title': '검색 제목 1',
                'link': 'https://example.com/1',
                'snippet': '발췌 1'
            },
            {
                'title': '검색 제목 2',
                'link': 'https://example.com/2',
                'snippet': '발췌 2'
            }
        ]
        
        # 검색 결과 스크래핑 테스트
        scraped_results = await self.retriever.scrape_search_results(search_results)
        
        # 검증
        self.assertEqual(len(scraped_results), 2)
        
        # 성공한 스크래핑
        self.assertEqual(scraped_results[0]['title'], '스크랩 제목 1')
        self.assertEqual(scraped_results[0]['content'], '스크랩 내용 1')
        self.assertEqual(scraped_results[0]['url'], 'https://example.com/1')
        
        # 실패한 스크래핑 (대체 데이터 사용)
        self.assertEqual(scraped_results[1]['title'], '검색 제목 2')
        self.assertEqual(scraped_results[1]['content'], '발췌 2')
        self.assertEqual(scraped_results[1]['url'], 'https://example.com/2')

# 비동기 테스트 실행 래퍼
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

# 테스트 메서드에 데코레이터 적용
TestWebRetriever.test_retrieve_documents = async_test(TestWebRetriever.test_retrieve_documents)
TestWebRetriever.test_search_and_scrape = async_test(TestWebRetriever.test_search_and_scrape)
TestWebRetriever.test_search_web = async_test(TestWebRetriever.test_search_web)
TestWebRetriever.test_scrape_search_results = async_test(TestWebRetriever.test_scrape_search_results)

if __name__ == '__main__':
    unittest.main() 
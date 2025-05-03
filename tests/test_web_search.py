#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
웹 검색 모듈 단위 테스트
"""

import unittest
import os
import sys
import logging
import json
from unittest import mock
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
from utils.web_search import MedicalWebSearch

class TestMedicalWebSearch(unittest.TestCase):
    """MedicalWebSearch 클래스 단위 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.api_key = os.getenv('GOOGLE_API_KEY', 'test_api_key')
        self.cse_id = os.getenv('GOOGLE_CSE_ID', 'test_cse_id')
        
        # 테스트용 웹 검색기 생성
        self.search = MedicalWebSearch(
            google_api_key=self.api_key,
            google_cse_id=self.cse_id
        )
    
    @patch('utils.web_search.build')
    def test_google_search_init(self, mock_build):
        """Google 검색 API 초기화 테스트"""
        # Mock 설정
        mock_build.return_value = MagicMock()
        
        # 새 인스턴스 생성하여 초기화 테스트
        search = MedicalWebSearch(
            google_api_key='test_key',
            google_cse_id='test_id'
        )
        
        # 초기화 확인
        mock_build.assert_called_once_with(
            'customsearch', 'v1', 
            developerKey='test_key',
            cache_discovery=False
        )
        
        self.assertIsNotNone(search.google_service)
    
    @patch('utils.web_search.build')
    def test_google_search(self, mock_build):
        """Google 검색 기능 테스트"""
        # Mock 응답 설정
        mock_result = {
            'items': [
                {
                    'title': '테스트 제목 1',
                    'link': 'https://example.com/1',
                    'snippet': '테스트 발췌 1'
                },
                {
                    'title': '테스트 제목 2',
                    'link': 'https://example.com/2',
                    'snippet': '테스트 발췌 2'
                }
            ]
        }
        
        # Mock 서비스 설정
        mock_service = MagicMock()
        mock_cse = MagicMock()
        mock_list = MagicMock()
        mock_execute = MagicMock(return_value=mock_result)
        
        mock_service.cse.return_value = mock_cse
        mock_cse.list.return_value = mock_list
        mock_list.execute.return_value = mock_execute.return_value
        
        mock_build.return_value = mock_service
        
        # 테스트용 웹 검색기 생성 (Mock 서비스 사용)
        search = MedicalWebSearch(
            google_api_key='test_key',
            google_cse_id='test_id'
        )
        search.google_service = mock_service
        
        # 검색 테스트
        results = search.google_search('테스트 쿼리', num_results=2)
        
        # 검색 호출 확인
        mock_cse.list.assert_called_once()
        mock_list.execute.assert_called_once()
        
        # 결과 확인
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], '테스트 제목 1')
        self.assertEqual(results[0]['link'], 'https://example.com/1')
        self.assertEqual(results[0]['snippet'], '테스트 발췌 1')
        self.assertEqual(results[0]['source'], 'Google')
    
    @patch('utils.web_search.DDGS')
    def test_duckduckgo_search(self, mock_ddgs):
        """DuckDuckGo 검색 기능 테스트"""
        # Mock 응답 설정
        mock_results = [
            {
                'title': 'DDG 테스트 제목 1',
                'href': 'https://example.org/1',
                'body': 'DDG 테스트 본문 1',
                'published': '2023-01-01'
            },
            {
                'title': 'DDG 테스트 제목 2',
                'href': 'https://example.org/2',
                'body': 'DDG 테스트 본문 2',
                'published': '2023-01-02'
            }
        ]
        
        # DDGS 컨텍스트 매니저 설정
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = mock_results
        
        # __enter__ 메서드가 mock_ddgs_instance를 반환하도록 설정
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
        
        # 검색 테스트
        results = self.search.duckduckgo_search('테스트 쿼리', num_results=2)
        
        # 검색 호출 확인
        mock_ddgs_instance.text.assert_called_once()
        
        # 결과 확인
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['title'], 'DDG 테스트 제목 1')
        self.assertEqual(results[0]['link'], 'https://example.org/1')
        self.assertEqual(results[0]['snippet'], 'DDG 테스트 본문 1')
        self.assertEqual(results[0]['source'], 'DuckDuckGo')
        self.assertEqual(results[0]['date'], '2023-01-01')
    
    @patch('utils.web_search.httpx.get')
    def test_bing_search(self, mock_get):
        """Bing 검색 기능 테스트"""
        # 환경 변수 보존
        original_key = os.environ.get('BING_SEARCH_KEY')
        os.environ['BING_SEARCH_KEY'] = 'test_bing_key'
        
        try:
            # Mock 응답 설정
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                'webPages': {
                    'value': [
                        {
                            'name': 'Bing 테스트 제목 1',
                            'url': 'https://example.net/1',
                            'snippet': 'Bing 테스트 발췌 1',
                            'dateLastCrawled': '2023-02-01'
                        },
                        {
                            'name': 'Bing 테스트 제목 2',
                            'url': 'https://example.net/2',
                            'snippet': 'Bing 테스트 발췌 2',
                            'dateLastCrawled': '2023-02-02'
                        }
                    ]
                }
            }
            mock_get.return_value = mock_response
            
            # 검색 테스트
            results = self.search.bing_search('테스트 쿼리', num_results=2)
            
            # 검색 호출 확인
            mock_get.assert_called_once()
            mock_response.raise_for_status.assert_called_once()
            mock_response.json.assert_called_once()
            
            # 결과 확인
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['title'], 'Bing 테스트 제목 1')
            self.assertEqual(results[0]['link'], 'https://example.net/1')
            self.assertEqual(results[0]['snippet'], 'Bing 테스트 발췌 1')
            self.assertEqual(results[0]['source'], 'Bing')
            self.assertEqual(results[0]['date'], '2023-02-01')
            
        finally:
            # 환경 변수 복원
            if original_key:
                os.environ['BING_SEARCH_KEY'] = original_key
            else:
                del os.environ['BING_SEARCH_KEY']
    
    @patch('utils.web_search.MedicalWebSearch.google_search')
    @patch('utils.web_search.MedicalWebSearch.duckduckgo_search')
    @patch('utils.web_search.MedicalWebSearch.bing_search')
    def test_search_integration(self, mock_bing, mock_ddg, mock_google):
        """통합 검색 기능 테스트"""
        # Mock 응답 설정
        mock_google.return_value = [{'title': 'Google 결과', 'source': 'Google'}]
        mock_ddg.return_value = [{'title': 'DuckDuckGo 결과', 'source': 'DuckDuckGo'}]
        mock_bing.return_value = [{'title': 'Bing 결과', 'source': 'Bing'}]
        
        # Google 검색 테스트 (기본 모드)
        self.search.google_service = MagicMock()  # Google 서비스 활성화
        results = self.search.search('테스트 쿼리', source='auto')
        mock_google.assert_called_once()
        self.assertEqual(results[0]['source'], 'Google')
        
        # DuckDuckGo 검색 테스트
        results = self.search.search('테스트 쿼리', source='duckduckgo')
        mock_ddg.assert_called_once()
        self.assertEqual(results[0]['source'], 'DuckDuckGo')
        
        # Bing 검색 테스트
        results = self.search.search('테스트 쿼리', source='bing')
        mock_bing.assert_called_once()
        self.assertEqual(results[0]['source'], 'Bing')
    
    def test_save_to_json(self):
        """JSON 저장 기능 테스트"""
        # 테스트 데이터
        test_results = [
            {
                'title': '테스트 제목 1',
                'link': 'https://example.com/1',
                'snippet': '테스트 발췌 1',
                'source': '테스트'
            }
        ]
        
        # 임시 파일명
        temp_filename = 'test_results.json'
        
        try:
            # JSON 저장 테스트
            saved_file = self.search.save_to_json(test_results, temp_filename)
            
            # 파일 존재 확인
            self.assertTrue(os.path.exists(temp_filename))
            
            # 파일 내용 확인
            with open(temp_filename, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(len(loaded_data), 1)
            self.assertEqual(loaded_data[0]['title'], '테스트 제목 1')
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

if __name__ == '__main__':
    unittest.main() 
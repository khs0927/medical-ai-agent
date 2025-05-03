#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
웹 스크래퍼 모듈 단위 테스트
"""

import unittest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 테스트 대상 모듈 가져오기
from utils.web_scraper import MedicalWebScraper

class TestMedicalWebScraper(unittest.TestCase):
    """MedicalWebScraper 클래스 단위 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.scraper = MedicalWebScraper()
    
    @patch('utils.web_scraper.requests.get')
    def test_fetch_url(self, mock_get):
        """URL 가져오기 테스트"""
        # Mock 응답 설정
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><h1>테스트 페이지</h1></body></html>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # URL 가져오기 테스트
        html_content = self.scraper._fetch_url('https://example.com')
        
        # 함수 호출 및 결과 확인
        mock_get.assert_called_once()
        self.assertEqual(html_content, '<html><body><h1>테스트 페이지</h1></body></html>')
    
    @patch('utils.web_scraper.requests.get')
    def test_fetch_url_error(self, mock_get):
        """URL 가져오기 오류 테스트"""
        # Mock 오류 응답 설정
        mock_get.side_effect = Exception('네트워크 오류')
        
        # URL 가져오기 테스트
        html_content = self.scraper._fetch_url('https://example.com')
        
        # 함수 호출 및 결과 확인
        mock_get.assert_called_once()
        self.assertIsNone(html_content)
    
    def test_parse_soup(self):
        """HTML 파싱 테스트"""
        # 테스트 HTML
        html = '<html><body><h1>테스트 제목</h1><p>테스트 내용</p></body></html>'
        
        # HTML 파싱 테스트
        soup = self.scraper._parse_soup(html)
        
        # 파싱 결과 확인
        self.assertIsInstance(soup, BeautifulSoup)
        self.assertEqual(soup.h1.text, '테스트 제목')
        self.assertEqual(soup.p.text, '테스트 내용')
    
    def test_clean_text(self):
        """텍스트 정리 테스트"""
        # 테스트 텍스트
        dirty_text = ' \n\t테스트   텍스트\n\n정리\t '
        
        # 텍스트 정리 테스트
        clean_text = self.scraper._clean_text(dirty_text)
        
        # 결과 확인
        self.assertEqual(clean_text, '테스트 텍스트 정리')
    
    @patch('utils.web_scraper.MedicalWebScraper._fetch_url')
    def test_scrape_pubmed_article(self, mock_fetch):
        """PubMed 논문 스크래핑 테스트"""
        # Mock HTML 설정
        mock_fetch.return_value = """
        <html>
            <head>
                <title>PubMed 테스트 논문</title>
                <meta name="citation_title" content="테스트 논문 제목">
                <meta name="citation_author" content="홍길동">
                <meta name="citation_author" content="김철수">
                <meta name="citation_journal_title" content="테스트 의학 저널">
                <meta name="citation_doi" content="10.1234/test.2023">
                <meta name="dc.date" content="2023-01-01">
            </head>
            <body>
                <div class="abstract-content">
                    <p>테스트 논문 초록입니다.</p>
                </div>
                <div class="keywords">
                    <span>키워드1</span>
                    <span>키워드2</span>
                </div>
            </body>
        </html>
        """
        
        # PubMed 스크래핑 테스트
        result = self.scraper.scrape_pubmed_article('https://pubmed.ncbi.nlm.nih.gov/12345/')
        
        # 결과 확인
        self.assertEqual(result['title'], '테스트 논문 제목')
        self.assertIn('홍길동', result['authors'])
        self.assertIn('김철수', result['authors'])
        self.assertEqual(result['journal'], '테스트 의학 저널')
        self.assertEqual(result['doi'], '10.1234/test.2023')
        self.assertEqual(result['abstract'], '테스트 논문 초록입니다.')
        self.assertIn('키워드1', result['keywords'])
        self.assertIn('키워드2', result['keywords'])
    
    @patch('utils.web_scraper.MedicalWebScraper._fetch_url')
    def test_scrape_nejm_article(self, mock_fetch):
        """NEJM 논문 스크래핑 테스트"""
        # Mock HTML 설정
        mock_fetch.return_value = """
        <html>
            <head>
                <title>NEJM 테스트 논문</title>
            </head>
            <body>
                <h1 class="title">NEJM 논문 제목</h1>
                <div class="authors">
                    <a>김의사</a>
                    <a>이간호</a>
                </div>
                <div class="doi">DOI: 10.1056/NEJM12345</div>
                <div class="article-header">
                    <p>2023년 1월 2일</p>
                </div>
                <div class="abstract">
                    <p>NEJM 논문 초록</p>
                </div>
            </body>
        </html>
        """
        
        # NEJM 스크래핑 테스트
        result = self.scraper.scrape_nejm_article('https://www.nejm.org/doi/full/12345')
        
        # 결과 확인
        self.assertEqual(result['title'], 'NEJM 논문 제목')
        self.assertIn('김의사', result['authors'])
        self.assertIn('이간호', result['authors'])
        self.assertEqual(result['journal'], 'The New England Journal of Medicine')
        self.assertEqual(result['doi'], '10.1056/NEJM12345')
        self.assertEqual(result['abstract'], 'NEJM 논문 초록')
    
    @patch('utils.web_scraper.MedicalWebScraper._fetch_url')
    def test_scrape_mayo_clinic(self, mock_fetch):
        """Mayo Clinic 페이지 스크래핑 테스트"""
        # Mock HTML 설정
        mock_fetch.return_value = """
        <html>
            <head>
                <title>Mayo Clinic - 테스트 질병</title>
            </head>
            <body>
                <h1>테스트 질병</h1>
                <div class="content">
                    <h2>증상</h2>
                    <p>증상 내용</p>
                    <h2>원인</h2>
                    <p>원인 내용</p>
                    <h2>치료</h2>
                    <p>치료 내용</p>
                </div>
            </body>
        </html>
        """
        
        # Mayo Clinic 스크래핑 테스트
        result = self.scraper.scrape_mayo_clinic('https://www.mayoclinic.org/diseases-conditions/test')
        
        # 결과 확인
        self.assertEqual(result['title'], '테스트 질병')
        self.assertEqual(result['source'], 'Mayo Clinic')
        self.assertIn('증상', result['content'])
        self.assertIn('증상 내용', result['content'])
        self.assertIn('원인', result['content'])
        self.assertIn('원인 내용', result['content'])
        self.assertIn('치료', result['content'])
        self.assertIn('치료 내용', result['content'])
    
    @patch('utils.web_scraper.MedicalWebScraper.scrape_pubmed_article')
    @patch('utils.web_scraper.MedicalWebScraper.scrape_nejm_article')
    @patch('utils.web_scraper.MedicalWebScraper.scrape_mayo_clinic')
    @patch('utils.web_scraper.MedicalWebScraper.scrape_generic_webpage')
    def test_scrape_medical_journal(self, mock_generic, mock_mayo, mock_nejm, mock_pubmed):
        """의학 저널 스크래핑 통합 테스트"""
        # Mock 응답 설정
        mock_pubmed.return_value = {'title': 'PubMed 논문'}
        mock_nejm.return_value = {'title': 'NEJM 논문'}
        mock_mayo.return_value = {'title': 'Mayo Clinic 페이지'}
        mock_generic.return_value = {'title': '일반 웹페이지'}
        
        # 여러 URL 스크래핑 테스트
        pubmed_result = self.scraper.scrape_medical_journal('https://pubmed.ncbi.nlm.nih.gov/12345/')
        nejm_result = self.scraper.scrape_medical_journal('https://www.nejm.org/doi/full/12345')
        mayo_result = self.scraper.scrape_medical_journal('https://www.mayoclinic.org/diseases-conditions/test')
        general_result = self.scraper.scrape_medical_journal('https://example.com')
        
        # 결과 확인
        self.assertEqual(pubmed_result['title'], 'PubMed 논문')
        self.assertEqual(nejm_result['title'], 'NEJM 논문')
        self.assertEqual(mayo_result['title'], 'Mayo Clinic 페이지')
        self.assertEqual(general_result['title'], '일반 웹페이지')
        
        mock_pubmed.assert_called_once()
        mock_nejm.assert_called_once()
        mock_mayo.assert_called_once()
        mock_generic.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
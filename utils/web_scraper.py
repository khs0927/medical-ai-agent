'''
의학 정보 수집을 위한 웹 스크래핑 모듈
'''

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from newspaper import Article
from requests.exceptions import RequestException

# 로깅 설정
logger = logging.getLogger(__name__)

class MedicalWebScraper:
    '''의학 웹사이트에서 정보를 스크래핑하는 클래스'''

    def __init__(self, timeout: int = 10, max_retries: int = 3, retry_delay: int = 2):
        '''스크래퍼 초기화

        Args:
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 지연 시간 (초)
        '''
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = UserAgent()
        self.session = requests.Session()
        
        # 스크래핑 결과 캐시
        self.cache = {}
        
        logger.info('의학 웹 스크래퍼 초기화 완료')

    def _get_headers(self) -> Dict[str, str]:
        '''랜덤 사용자 에이전트로 헤더 생성'''
        return {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',  # Do Not Track 요청
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def _get_page_content(self, url: str) -> Optional[str]:
        '''URL에서 페이지 내용 가져오기 (재시도 로직 포함)'''
        # 캐시 확인
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in self.cache:
            logger.debug(f'캐시에서 {url} 로드')
            return self.cache[cache_key]
            
        attempts = 0
        while attempts < self.max_retries:
            try:
                logger.debug(f'웹페이지 요청: {url}')
                headers = self._get_headers()
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                
                # 상태 코드 확인
                if response.status_code == 200:
                    content = response.text
                    
                    # 캐시에 저장
                    self.cache[cache_key] = content
                    
                    return content
                elif response.status_code == 404:
                    logger.warning(f'페이지를 찾을 수 없음: {url}')
                    return None
                else:
                    logger.warning(f'HTTP 오류 {response.status_code}: {url}')
            
            except RequestException as e:
                logger.warning(f'요청 실패 ({attempts+1}/{self.max_retries}): {url} - {e}')
            
            # 재시도 전 대기
            attempts += 1
            if attempts < self.max_retries:
                time.sleep(self.retry_delay)
        
        logger.error(f'최대 재시도 횟수 초과: {url}')
        return None

    def _parse_html(self, html_content: str) -> BeautifulSoup:
        '''HTML 내용을 파싱하여 BeautifulSoup 객체 반환'''
        return BeautifulSoup(html_content, 'lxml')

    def scrape_article(self, url: str) -> Dict[str, Any]:
        '''뉴스 또는 의학 아티클 스크래핑

        Args:
            url: 아티클 URL

        Returns:
            아티클 정보(제목, 내용, 저자 등)를 포함한 딕셔너리
        '''
        try:
            logger.info(f"웹 페이지 스크랩 시작: {url}")
            
            # newspaper3k 사용
            article = Article(url)
            article.download()
            article.parse()
            
            # 필요시 NLP 처리
            article.nlp()
            
            # 결과 반환
            result = {
                'url': url,
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'keywords': article.keywords,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image,
                'metadata': {
                    'source_url': url,
                    'domain': urlparse(url).netloc,
                    'scrape_time': time.time(),
                }
            }
            
            logger.info(f'$1\"$2\"$3')
            return result
            
        except Exception as e:
            logger.error(f'아티클 스크래핑 실패: {url} - {e}', exc_info=True)
            # 기본 스크래핑으로 폴백
            return self.scrape_generic_page(url)

    def scrape_generic_page(self, url: str) -> Dict[str, Any]:
        '''일반 웹페이지 스크래핑

        Args:
            url: 웹페이지 URL

        Returns:
            페이지 정보를 포함한 딕셔너리
        '''
        logger.info(f'일반 페이지 스크래핑: {url}')
        
        content = self._get_page_content(url)
        if not content:
            return {'url': url, 'error': '콘텐츠를 가져올 수 없습니다'}
        
        soup = self._parse_html(content)
        
        # 제목 추출
        title = None
        if soup.title:
            title = soup.title.string
        
        # 메타 설명 추출
        meta_description = None
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag and 'content' in meta_tag.attrs:
            meta_description = meta_tag['content']
        
        # 본문 내용 추출 (다양한 일반적인 콘텐츠 컨테이너)
        content_selectors = [
            'article', 'main', '.content', '#content',
            '.post', '.entry', '.article-content',
            '[role=\'main\']', '.post-content', '.entry-content'
        ]
        
        main_content = ''
        for selector in content_selectors:
            content_tag = soup.select_one(selector)
            if content_tag:
                # 스크립트와 스타일 제거
                for script in content_tag(['script', 'style']):
                    script.decompose()
                
                # 텍스트 추출 및 정리
                text = content_tag.get_text(separator='\n').strip()
                # 여러 개의 공백 처리
                text = re.sub(r'\n+', '\n', text)
                text = re.sub(r'\s+', ' ', text)
                
                main_content = text
                break
        
        # 콘텐츠를 찾지 못한 경우, 전체 본문에서 추출
        if not main_content:
            # 스크립트와 스타일 제거
            for script in soup(['script', 'style', 'header', 'footer', 'nav']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            text = soup.get_text(separator='\n').strip()
            # 여러 개의 공백 처리
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            main_content = text
        
        # 결과 반환
        result = {
            'url': url,
            'title': title,
            'meta_description': meta_description,
            'content': main_content,
            'html': content,  # 원본 HTML
            'metadata': {
                'source_url': url,
                'domain': urlparse(url).netloc,
                'scrape_time': time.time(),
            }
        }
        
        logger.info(f'일반 페이지 스크래핑 성공: {url}')
        return result

    def scrape_medical_journal(self, url: str) -> Dict[str, Any]:
        '''의학 저널 아티클 스크래핑 (특화된 파싱)

        Args:
            url: 의학 저널 URL

        Returns:
            저널 아티클 정보를 포함한 딕셔너리
        '''
        logger.info(f'의학 저널 스크래핑: {url}')
        
        content = self._get_page_content(url)
        if not content:
            return {'url': url, 'error': '콘텐츠를 가져올 수 없습니다'}
        
        soup = self._parse_html(content)
        domain = urlparse(url).netloc
        
        # 저널별 특화 파싱
        if 'nejm.org' in domain:
            return self._parse_nejm(soup, url)
        elif 'jamanetwork.com' in domain:
            return self._parse_jama(soup, url)
        elif 'thelancet.com' in domain:
            return self._parse_lancet(soup, url)
        elif 'bmj.com' in domain:
            return self._parse_bmj(soup, url)
        elif 'sciencedirect.com' in domain:
            return self._parse_sciencedirect(soup, url)
        elif 'pubmed.ncbi.nlm.nih.gov' in domain:
            return self._parse_pubmed(soup, url)
        else:
            # 기본 아티클 파싱으로 폴백
            logger.info(f'특화 파싱 없는 저널, 일반 파싱 사용: {domain}')
            return self.scrape_article(url)

    def _parse_nejm(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''New England Journal of Medicine 아티클 파싱'''
        title = soup.select_one('h1.title')
        title_text = title.get_text().strip() if title else None
        
        # 저자 추출
        authors_div = soup.select_one('div.authors')
        authors = []
        if authors_div:
            author_links = authors_div.select('a.author-name')
            for author in author_links:
                authors.append(author.get_text().strip())
        
        # 초록 추출
        abstract = ''
        abstract_div = soup.select_one('div#abstract')
        if abstract_div:
            abstract = abstract_div.get_text().strip()
        
        # 본문 추출
        article_body = soup.select_one('div#article_body')
        content = ''
        if article_body:
            # 스크립트와 스타일 제거
            for script in article_body(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_body.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # DOI 추출
        doi = None
        doi_elem = soup.select_one('meta[name=\'doi\']')
        if doi_elem and 'content' in doi_elem.attrs:
            doi = doi_elem['content']
        
        # 발행일 추출
        pub_date = None
        date_elem = soup.select_one('meta[name=\'article:published_time\']')
        if date_elem and 'content' in date_elem.attrs:
            pub_date = date_elem['content']
        
        return {
            'url': url,
            'title': title_text,
            'authors': authors,
            'abstract': abstract,
            'content': content,
            'doi': doi,
            'publication_date': pub_date,
            'journal': 'New England Journal of Medicine',
            'metadata': {
                'source_url': url,
                'domain': 'nejm.org',
                'scrape_time': time.time(),
            }
        }

    def _parse_jama(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''JAMA Network 아티클 파싱'''
        title = soup.select_one('h1.article-title')
        title_text = title.get_text().strip() if title else None
        
        # 저자 추출
        authors = []
        author_list = soup.select('div.article-author')
        for author in author_list:
            authors.append(author.get_text().strip())
        
        # 초록 추출
        abstract = ''
        abstract_div = soup.select_one('div.abstract')
        if abstract_div:
            abstract = abstract_div.get_text().strip()
        
        # 본문 추출
        article_body = soup.select_one('div.article-full-text')
        content = ''
        if article_body:
            # 스크립트와 스타일 제거
            for script in article_body(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_body.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # DOI 추출
        doi = None
        doi_elem = soup.select_one('meta[name=\'citation_doi\']')
        if doi_elem and 'content' in doi_elem.attrs:
            doi = doi_elem['content']
        
        # 발행일 추출
        pub_date = None
        date_elem = soup.select_one('meta[name=\'citation_publication_date\']')
        if date_elem and 'content' in date_elem.attrs:
            pub_date = date_elem['content']
        
        return {
            'url': url,
            'title': title_text,
            'authors': authors,
            'abstract': abstract,
            'content': content,
            'doi': doi,
            'publication_date': pub_date,
            'journal': 'JAMA Network',
            'metadata': {
                'source_url': url,
                'domain': 'jamanetwork.com',
                'scrape_time': time.time(),
            }
        }

    def _parse_lancet(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''The Lancet 아티클 파싱'''
        # Lancet 사이트에 맞는 파싱 로직 구현
        title = soup.select_one('h1.article-header__title')
        title_text = title.get_text().strip() if title else None
        
        # 저자 추출
        authors = []
        author_list = soup.select('a.loa__item-name')
        for author in author_list:
            authors.append(author.get_text().strip())
        
        # 초록 추출
        abstract = ''
        abstract_div = soup.select_one('div.section-paragraph')
        if abstract_div:
            abstract = abstract_div.get_text().strip()
        
        # 본문 추출
        article_body = soup.select_one('div.article-body')
        content = ''
        if article_body:
            # 스크립트와 스타일 제거
            for script in article_body(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_body.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # DOI 추출
        doi = None
        doi_elem = soup.select_one('a.doi')
        if doi_elem:
            doi_text = doi_elem.get_text().strip()
            doi_match = re.search(r'10\.\d{4}/\S+', doi_text)
            if doi_match:
                doi = doi_match.group(0)
        
        # 발행일 추출
        pub_date = None
        date_elem = soup.select_one('div.article-header__date')
        if date_elem:
            pub_date = date_elem.get_text().strip()
        
        return {
            'url': url,
            'title': title_text,
            'authors': authors,
            'abstract': abstract,
            'content': content,
            'doi': doi,
            'publication_date': pub_date,
            'journal': 'The Lancet',
            'metadata': {
                'source_url': url,
                'domain': 'thelancet.com',
                'scrape_time': time.time(),
            }
        }
    
    def _parse_bmj(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''BMJ (British Medical Journal) 아티클 파싱'''
        title = soup.select_one('h1.highwire-cite-title')
        title_text = title.get_text().strip() if title else None
        
        # 저자 추출
        authors = []
        author_list = soup.select('div.highwire-cite-authors a')
        for author in author_list:
            authors.append(author.get_text().strip())
        
        # 초록 추출
        abstract = ''
        abstract_div = soup.select_one('div.abstract')
        if abstract_div:
            abstract = abstract_div.get_text().strip()
        
        # 본문 추출
        article_body = soup.select_one('div.article-body')
        content = ''
        if article_body:
            # 스크립트와 스타일 제거
            for script in article_body(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_body.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # DOI 추출
        doi = None
        doi_elem = soup.select_one('meta[name=\'citation_doi\']')
        if doi_elem and 'content' in doi_elem.attrs:
            doi = doi_elem['content']
        
        # 발행일 추출
        pub_date = None
        date_elem = soup.select_one('meta[name=\'citation_publication_date\']')
        if date_elem and 'content' in date_elem.attrs:
            pub_date = date_elem['content']
        
        return {
            'url': url,
            'title': title_text,
            'authors': authors,
            'abstract': abstract,
            'content': content,
            'doi': doi,
            'publication_date': pub_date,
            'journal': 'BMJ',
            'metadata': {
                'source_url': url,
                'domain': 'bmj.com',
                'scrape_time': time.time(),
            }
        }
    
    def _parse_sciencedirect(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''ScienceDirect 아티클 파싱'''
        title = soup.select_one('h1.title-text')
        title_text = title.get_text().strip() if title else None
        
        # 저자 추출
        authors = []
        author_list = soup.select('a.author')
        for author in author_list:
            authors.append(author.get_text().strip())
        
        # 초록 추출
        abstract = ''
        abstract_div = soup.select_one('div.abstract')
        if abstract_div:
            abstract = abstract_div.get_text().strip()
        
        # 본문 추출 (ScienceDirect는 제한된 접근인 경우가 많음)
        article_body = soup.select_one('div.article-body')
        content = ''
        if article_body:
            # 스크립트와 스타일 제거
            for script in article_body(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_body.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # DOI 추출
        doi = None
        doi_elem = soup.select_one('a.doi')
        if doi_elem:
            doi_text = doi_elem.get_text().strip()
            doi_match = re.search(r'10\.\d{4}/\S+', doi_text)
            if doi_match:
                doi = doi_match.group(0)
        
        # 발행일 추출
        pub_date = None
        date_elem = soup.select_one('div.publication-date')
        if date_elem:
            pub_date = date_elem.get_text().strip()
        
        # 저널명 추출
        journal = None
        journal_elem = soup.select_one('a.publication-title-link')
        if journal_elem:
            journal = journal_elem.get_text().strip()
        
        return {
            'url': url,
            'title': title_text,
            'authors': authors,
            'abstract': abstract,
            'content': content,
            'doi': doi,
            'publication_date': pub_date,
            'journal': journal,
            'metadata': {
                'source_url': url,
                'domain': 'sciencedirect.com',
                'scrape_time': time.time(),
            }
        }
    
    def _parse_pubmed(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''PubMed 아티클 파싱'''
        title = soup.select_one('h1.heading-title')
        title_text = title.get_text().strip() if title else None
        
        # 저자 추출
        authors = []
        author_list = soup.select('span.authors-list-item')
        for author in author_list:
            authors.append(author.get_text().strip())
        
        # 초록 추출
        abstract = ''
        abstract_div = soup.select_one('div#abstract')
        if abstract_div:
            abstract = abstract_div.get_text().strip()
        
        # DOI 추출
        doi = None
        doi_elem = soup.select_one('span.identifier.doi')
        if doi_elem:
            doi_text = doi_elem.get_text().strip()
            doi_match = re.search(r'10\.\d{4}/\S+', doi_text)
            if doi_match:
                doi = doi_match.group(0)
        
        # PMID 추출
        pmid = None
        pmid_elem = soup.select_one('span.identifier.pubmed')
        if pmid_elem:
            pmid_text = pmid_elem.get_text().strip()
            pmid_match = re.search(r'\d+', pmid_text)
            if pmid_match:
                pmid = pmid_match.group(0)
        
        # 발행일 추출
        pub_date = None
        date_elem = soup.select_one('span.cit')
        if date_elem:
            pub_date = date_elem.get_text().strip()
        
        # 저널명 추출
        journal = None
        journal_elem = soup.select_one('button.journal-actions-trigger')
        if journal_elem:
            journal = journal_elem.get_text().strip()
        
        return {
            'url': url,
            'title': title_text,
            'authors': authors,
            'abstract': abstract,
            'content': abstract,  # PubMed는 보통 초록만 제공
            'doi': doi,
            'pmid': pmid,
            'publication_date': pub_date,
            'journal': journal,
            'metadata': {
                'source_url': url,
                'domain': 'pubmed.ncbi.nlm.nih.gov',
                'scrape_time': time.time(),
            }
        }

    def scrape_medical_guidelines(self, url: str) -> Dict[str, Any]:
        '''의학 가이드라인 스크래핑 (특화된 파싱)

        Args:
            url: 의학 가이드라인 URL

        Returns:
            가이드라인 정보를 포함한 딕셔너리
        '''
        logger.info(f'의학 가이드라인 스크래핑: {url}')
        
        content = self._get_page_content(url)
        if not content:
            return {'url': url, 'error': '콘텐츠를 가져올 수 없습니다'}
        
        soup = self._parse_html(content)
        domain = urlparse(url).netloc
        
        # 가이드라인 사이트별 특화 파싱
        if 'guidelines.gov' in domain or 'ahrq.gov' in domain:
            return self._parse_ngc_guidelines(soup, url)
        elif 'nice.org.uk' in domain:
            return self._parse_nice_guidelines(soup, url)
        elif 'acc.org' in domain:
            return self._parse_acc_guidelines(soup, url)
        else:
            # 기본 아티클 파싱으로 폴백
            logger.info(f'특화 파싱 없는 가이드라인, 일반 파싱 사용: {domain}')
            return self.scrape_article(url)

    def _parse_ngc_guidelines(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''National Guideline Clearinghouse 가이드라인 파싱'''
        title = soup.select_one('h1.guideline-title')
        title_text = title.get_text().strip() if title else None
        
        # 조직 추출
        organization = None
        org_div = soup.select_one('div.guideline-developer')
        if org_div:
            organization = org_div.get_text().strip()
        
        # 발행일 추출
        pub_date = None
        date_div = soup.select_one('div.guideline-released')
        if date_div:
            pub_date = date_div.get_text().strip()
        
        # 본문 추출
        content = ''
        summary_div = soup.select_one('div.guideline-summary')
        if summary_div:
            # 스크립트와 스타일 제거
            for script in summary_div(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = summary_div.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # 추천 수준 추출
        recommendation_levels = []
        rec_divs = soup.select('div.recommendation-strength')
        for rec in rec_divs:
            recommendation_levels.append(rec.get_text().strip())
        
        return {
            'url': url,
            'title': title_text,
            'organization': organization,
            'publication_date': pub_date,
            'content': content,
            'recommendation_levels': recommendation_levels,
            'metadata': {
                'source_url': url,
                'domain': urlparse(url).netloc,
                'scrape_time': time.time(),
            }
        }

    def _parse_nice_guidelines(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''NICE (영국 국립보건임상연구원) 가이드라인 파싱'''
        title = soup.select_one('h1.page-header__heading')
        title_text = title.get_text().strip() if title else None
        
        # 발행일 추출
        pub_date = None
        date_div = soup.select_one('p.published-date')
        if date_div:
            pub_date = date_div.get_text().strip()
        
        # 본문 추출
        content = ''
        article_div = soup.select_one('div.chapter')
        if article_div:
            # 스크립트와 스타일 제거
            for script in article_div(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_div.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        # 추천 내용 추출
        recommendations = []
        rec_divs = soup.select('div.recommendation')
        for rec in rec_divs:
            rec_text = rec.get_text().strip()
            if rec_text:
                recommendations.append(rec_text)
        
        return {
            'url': url,
            'title': title_text,
            'organization': 'NICE (National Institute for Health and Care Excellence)',
            'publication_date': pub_date,
            'content': content,
            'recommendations': recommendations,
            'metadata': {
                'source_url': url,
                'domain': 'nice.org.uk',
                'scrape_time': time.time(),
            }
        }

    def _parse_acc_guidelines(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        '''American College of Cardiology 가이드라인 파싱'''
        title = soup.select_one('h1.page-title')
        title_text = title.get_text().strip() if title else None
        
        # 발행일 추출
        pub_date = None
        date_div = soup.select_one('span.date-display-single')
        if date_div:
            pub_date = date_div.get_text().strip()
        
        # 저자 추출
        authors = []
        author_divs = soup.select('div.field-name-field-author div.field-item')
        for author in author_divs:
            authors.append(author.get_text().strip())
        
        # 본문 추출
        content = ''
        article_div = soup.select_one('div.field-name-body')
        if article_div:
            # 스크립트와 스타일 제거
            for script in article_div(['script', 'style']):
                script.decompose()
            
            # 텍스트 추출 및 정리
            content = article_div.get_text(separator='\n').strip()
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r'\s+', ' ', content)
        
        return {
            'url': url,
            'title': title_text,
            'organization': 'American College of Cardiology',
            'authors': authors,
            'publication_date': pub_date,
            'content': content,
            'metadata': {
                'source_url': url,
                'domain': 'acc.org',
                'scrape_time': time.time(),
            }
        }

    def save_to_json(self, data: Dict[str, Any], filename: str) -> str:
        '''스크래핑 데이터를 JSON 파일로 저장

        Args:
            data: 저장할 데이터
            filename: 파일명

        Returns:
            저장된 파일 경로
        '''
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f'데이터가 {filename}에 저장되었습니다.')
            return filename
        
        except Exception as e:
            logger.error(f'JSON 저장 실패: {e}', exc_info=True)
            return '' 
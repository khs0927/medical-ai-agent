"""
의학 정보 검색을 위한 웹 검색 모듈
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 로깅 설정
logger = logging.getLogger(__name__)


class MedicalWebSearch:
    """의학 정보 검색을 위한 웹 검색 클래스"""

    def __init__(self, 
                 google_api_key: Optional[str] = None, 
                 google_cse_id: Optional[str] = None,
                 timeout: int = 10, 
                 max_retries: int = 3, 
                 retry_delay: int = 2):
        """웹 검색 초기화

        Args:
            google_api_key: Google Custom Search API 키
            google_cse_id: Google Custom Search Engine ID
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간 지연 시간 (초)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Google API 설정
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = google_cse_id or os.getenv('GOOGLE_CSE_ID')
        self.google_service = None
        
        if self.google_api_key and self.google_cse_id:
            try:
                self.google_service = build(
                    'customsearch', 'v1', 
                    developerKey=self.google_api_key,
                    cache_discovery=False
                )
                logger.info('Google 검색 API 초기화 완료')
            except Exception as e:
                logger.error(f'Google 검색 API 초기화 오류: {e}', exc_info=True)
        
        # 검색 결과 캐시
        self.cache = {}
        
        logger.info('의학 웹 검색 초기화 완료')

    def google_search(self, query: str, num_results: int = 10, 
                      lang: str = 'ko', country: str = 'kr',
                      medical_site_filter: bool = True) -> List[Dict[str, str]]:
        """Google 커스텀 검색 수행

        Args:
            query: 검색 쿼리
            num_results: 결과 수
            lang: 검색 언어
            country: 검색 국가
            medical_site_filter: 의학 사이트로 필터링할지 여부

        Returns:
            검색 결과 목록
        """
        # 캐시 확인
        cache_key = f'google_{query}_{num_results}_{lang}_{country}_{medical_site_filter}'
        if cache_key in self.cache:
            logger.debug(f'캐시에서 Google 검색 결과 로드: {query}')
            return self.cache[cache_key]
        
        # 의학 관련 쿼리 보완
        if medical_site_filter:
            medical_sites = [
                'pubmed.ncbi.nlm.nih.gov', 'nih.gov', 'nejm.org', 
                'who.int', 'mayoclinic.org', 'jamanetwork.com',
                'thelancet.com', 'bmj.com', 'medscape.com',
                'health.harvard.edu', 'webmd.com', 'cdc.gov', 
                'medlineplus.gov', 'aafp.org', 'ahajournals.org',
                'acc.org', 'cancer.gov', 'diabetes.org', 'healthline.com',
                'uptodate.com', 'cochrane.org', 'nature.com/nm', 
                'sciencedirect.com', 'nice.org.uk', 'ahrq.gov', 
                'kdca.go.kr', 'mohw.go.kr', 'snuh.org', 'khmc.or.kr'
            ]
            # OR 연산자로 사이트 필터 추가
            site_filter = ' OR '.join([f'site:{site}' for site in medical_sites[:10]])
            enhanced_query = f'({query}) ({site_filter})'
        else:
            enhanced_query = query
        
        results = []
        
        if not self.google_service:
            logger.warning('Google 검색 API가 초기화되지 않았습니다. DuckDuckGo 검색으로 폴백합니다.')
            return self.duckduckgo_search(query, num_results)
        
        try:
            # Google Custom Search API 호출
            search_results = self.google_service.cse().list(
                q=enhanced_query,
                cx=self.google_cse_id,
                num=min(num_results, 10),  # Google API는 한 번에 최대 10개 결과
                lr=f'lang_{lang}',
                gl=country.upper(),
                safe='active'
            ).execute()
            
            # 결과 파싱
            if 'items' in search_results:
                for item in search_results['items']:
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'Google',
                        'date': '',  # Google API는 날짜를 제공하지 않음
                    }
                    
                    # 날짜 추출 시도
                    if 'pagemap' in item and 'metatags' in item['pagemap']:
                        for metatag in item['pagemap']['metatags']:
                            if 'article:published_time' in metatag:
                                result['date'] = metatag['article:published_time']
                                break
                    
                    results.append(result)
            
            # 결과가 10개 미만이고 요청된 결과가 10개 이상인 경우, 추가 페이지 조회
            start_index = 11  # Google API에서 두 번째 페이지 시작 인덱스
            while len(results) < num_results and 'nextPage' in search_results.get('queries', {}):
                search_results = self.google_service.cse().list(
                    q=enhanced_query,
                    cx=self.google_cse_id,
                    num=min(num_results - len(results), 10),
                    lr=f'lang_{lang}',
                    gl=country.upper(),
                    safe='active',
                    start=start_index
                ).execute()
                
                if 'items' in search_results:
                    for item in search_results['items']:
                        result = {
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'source': 'Google',
                            'date': '',
                        }
                        
                        if 'pagemap' in item and 'metatags' in item['pagemap']:
                            for metatag in item['pagemap']['metatags']:
                                if 'article:published_time' in metatag:
                                    result['date'] = metatag['article:published_time']
                                    break
                        
                        results.append(result)
                
                start_index += 10
            
            # 캐시에 저장
            self.cache[cache_key] = results
            
            logger.info(f'Google 검색 완료: {query}, {len(results)}개 결과')
            return results
            
        except HttpError as e:
            logger.error(f'Google 검색 API 오류: {e}', exc_info=True)
            # DuckDuckGo로 폴백
            logger.info('DuckDuckGo 검색으로 폴백')
            return self.duckduckgo_search(query, num_results)
            
        except Exception as e:
            logger.error(f'Google 검색 오류: {e}', exc_info=True)
            return []

    def duckduckgo_search(self, query: str, num_results: int = 10, 
                         region: str = 'kr-kr', 
                         medical_filter: bool = True) -> List[Dict[str, str]]:
        """DuckDuckGo 검색 수행

        Args:
            query: 검색 쿼리
            num_results: 결과 수
            region: 검색 지역
            medical_filter: 의학 키워드 필터 적용 여부

        Returns:
            검색 결과 목록
        """
        # 캐시 확인
        cache_key = f'ddg_{query}_{num_results}_{region}_{medical_filter}'
        if cache_key in self.cache:
            logger.debug(f'캐시에서 DuckDuckGo 검색 결과 로드: {query}')
            return self.cache[cache_key]
        
        # 의학 필터 적용
        if medical_filter:
            medical_terms = ['medical', 'health', '의학', '건강', 'clinical', '임상']
            enhanced_query = f"{query} {' '.join(medical_terms[:3])}"
        else:
            enhanced_query = query
        
        results = []
        attempts = 0
        
        while attempts < self.max_retries and len(results) < num_results:
            try:
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.text(
                        enhanced_query, 
                        region=region, 
                        safesearch='on', 
                        timelimit='y',  # 최근 1년
                        max_results=num_results
                    ))
                
                for item in ddg_results:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('href', ''),
                        'snippet': item.get('body', ''),
                        'source': 'DuckDuckGo',
                        'date': item.get('published', '')
                    })
                
                break
                
            except Exception as e:
                logger.warning(f'DuckDuckGo 검색 시도 실패 ({attempts+1}/{self.max_retries}): {e}')
                attempts += 1
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # 결과 정렬 및 제한
        results = results[:num_results]
        
        # 캐시에 저장
        self.cache[cache_key] = results
        
        logger.info(f'DuckDuckGo 검색 완료: {query}, {len(results)}개 결과')
        return results

    def bing_search(self, query: str, num_results: int = 10, 
                   mkt: str = 'ko-KR', 
                   subscription_key: Optional[str] = None) -> List[Dict[str, str]]:
        """Bing 검색 수행

        Args:
            query: 검색 쿼리
            num_results: 결과 수
            mkt: 시장 코드
            subscription_key: Bing API 구독 키

        Returns:
            검색 결과 목록
        """
        # 캐시 확인
        cache_key = f'bing_{query}_{num_results}_{mkt}'
        if cache_key in self.cache:
            logger.debug(f'캐시에서 Bing 검색 결과 로드: {query}')
            return self.cache[cache_key]
        
        subscription_key = subscription_key or os.getenv('BING_SEARCH_KEY')
        if not subscription_key:
            logger.warning('Bing API 키가 제공되지 않았습니다. DuckDuckGo 검색으로 폴백합니다.')
            return self.duckduckgo_search(query, num_results)
        
        search_url = 'https://api.bing.microsoft.com/v7.0/search'
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        params = {
            'q': query,
            'count': min(num_results, 50),  # Bing API는 한 번에 최대 50개 결과
            'mkt': mkt,
            'responseFilter': 'Webpages',
            'safeSearch': 'Moderate'
        }
        
        results = []
        attempts = 0
        
        while attempts < self.max_retries:
            try:
                response = httpx.get(
                    search_url, 
                    headers=headers, 
                    params=params, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                search_results = response.json()
                
                if 'webPages' in search_results and 'value' in search_results['webPages']:
                    for item in search_results['webPages']['value']:
                        results.append({
                            'title': item.get('name', ''),
                            'link': item.get('url', ''),
                            'snippet': item.get('snippet', ''),
                            'source': 'Bing',
                            'date': item.get('dateLastCrawled', '')
                        })
                
                break
                
            except Exception as e:
                logger.warning(f'Bing 검색 시도 실패 ({attempts+1}/{self.max_retries}): {e}')
                attempts += 1
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # 캐시에 저장
        self.cache[cache_key] = results
        
        logger.info(f'Bing 검색 완료: {query}, {len(results)}개 결과')
        return results

    def search(self, query: str, num_results: int = 10, source: str = 'auto') -> List[Dict[str, str]]:
        """통합 검색 수행

        Args:
            query: 검색 쿼리
            num_results: 결과 수
            source: 검색 소스 ("google", "duckduckgo", "bing", "auto")

        Returns:
            검색 결과 목록
        """
        logger.info(f"'{query}' 검색")
        
        if source == 'google' or (source == 'auto' and self.google_service):
            return self.google_search(query, num_results)
        elif source == 'bing' or (source == 'auto' and os.getenv('BING_SEARCH_KEY')):
            return self.bing_search(query, num_results)
        else:
            return self.duckduckgo_search(query, num_results)

    def search_medical_info(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """의학 정보 특화 검색

        Args:
            query: 검색 쿼리
            num_results: 결과 수

        Returns:
            검색 결과 목록
        """
        # 의학 용어 쿼리 보완
        if not any(term in query.lower() for term in ['medical', 'health', 'clinical', '의학', '건강', '임상']):
            enhanced_query = f'{query} 의학 정보'
        else:
            enhanced_query = query
            
        # 우선순위 순으로 검색
        if self.google_service:
            results = self.google_search(enhanced_query, num_results, medical_site_filter=True)
        elif os.getenv('BING_SEARCH_KEY'):
            results = self.bing_search(enhanced_query, num_results)
        else:
            results = self.duckduckgo_search(enhanced_query, num_results, medical_filter=True)
            
        return results

    def search_medical_journals(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """의학 저널 특화 검색

        Args:
            query: 검색 쿼리
            num_results: 결과 수

        Returns:
            검색 결과 목록
        """
        journals = [
            'pubmed.ncbi.nlm.nih.gov', 'nejm.org', 'jamanetwork.com', 
            'thelancet.com', 'bmj.com', 'sciencedirect.com', 'nature.com/nm'
        ]
        
        # 사이트 제한 쿼리 생성
        journal_filter = ' OR '.join([f'site:{journal}' for journal in journals])
        enhanced_query = f'{query} ({journal_filter})'
        
        # 검색 수행
        if self.google_service:
            results = self.google_search(enhanced_query, num_results, medical_site_filter=False)
        else:
            # DuckDuckGo는 site: 필터를 지원하지만 OR 연산자를 제대로 처리하지 않음
            # 각 저널마다 개별 검색 후 병합
            all_results = []
            results_per_journal = max(num_results // len(journals), 2)
            
            for journal in journals:
                with DDGS() as ddgs:
                    journal_query = f'{query} site:{journal}'
                    try:
                        journal_results = list(ddgs.text(
                            journal_query, 
                            region='wt-wt',  # global
                            safesearch='on',
                            max_results=results_per_journal
                        ))
                        
                        for item in journal_results:
                            all_results.append({
                                'title': item.get('title', ''),
                                'link': item.get('href', ''),
                                'snippet': item.get('body', ''),
                                'source': f'DuckDuckGo ({journal})',
                                'date': item.get('published', '')
                            })
                    except Exception as e:
                        logger.warning(f'{journal} 검색 오류: {e}')
            
            # 결과 정렬 및 제한
            results = all_results[:num_results]
            
        logger.info(f'의학 저널 검색 완료: {query}, {len(results)}개 결과')
        return results

    def search_clinical_guidelines(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """임상 가이드라인 특화 검색

        Args:
            query: 검색 쿼리
            num_results: 결과 수

        Returns:
            검색 결과 목록
        """
        guideline_sites = [
            'guidelines.gov', 'nice.org.uk', 'ahrq.gov', 'kdca.go.kr', 
            'guideline.or.kr', 'acc.org/guidelines', 'heartfoundation.org.nz/guidelines',
            'who.int/publications/guidelines'
        ]
        
        # 사이트 제한 쿼리 생성
        site_filter = ' OR '.join([f'site:{site}' for site in guideline_sites])
        # 가이드라인 관련 키워드 추가
        keywords = ['guideline', 'recommendation', 'clinical practice', '가이드라인', '지침', '임상지침']
        keyword_filter = ' OR '.join(keywords)
        enhanced_query = f'{query} ({site_filter}) ({keyword_filter})'
        
        # 검색 수행
        if self.google_service:
            results = self.google_search(enhanced_query, num_results, medical_site_filter=False)
        else:
            # 각 사이트별 개별 검색
            all_results = []
            results_per_site = max(num_results // len(guideline_sites), 1)
            
            for site in guideline_sites:
                with DDGS() as ddgs:
                    site_query = f'{query} site:{site} guideline'
                    try:
                        site_results = list(ddgs.text(
                            site_query, 
                            region='wt-wt',
                            safesearch='on',
                            max_results=results_per_site
                        ))
                        
                        for item in site_results:
                            all_results.append({
                                'title': item.get('title', ''),
                                'link': item.get('href', ''),
                                'snippet': item.get('body', ''),
                                'source': f'DuckDuckGo ({site})',
                                'date': item.get('published', '')
                            })
                    except Exception as e:
                        logger.warning(f'{site} 검색 오류: {e}')
            
            # 결과 정렬 및 제한
            results = all_results[:num_results]
            
        logger.info(f'임상 가이드라인 검색 완료: {query}, {len(results)}개 결과')
        return results

    def save_to_json(self, results: List[Dict[str, str]], filename: str) -> str:
        """검색 결과를 JSON 파일로 저장

        Args:
            results: 검색 결과
            filename: 파일명

        Returns:
            저장된 파일 경로
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f'검색 결과가 {filename}에 저장되었습니다.')
            return filename
        
        except Exception as e:
            logger.error(f'JSON 저장 실패: {e}', exc_info=True)
            return '' 
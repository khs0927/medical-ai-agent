#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
의료 AI 에이전트의 웹 검색 및 스크래핑 통합 테스트
"""

import asyncio
import os
import sys
import logging
import json
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 환경 변수 로드
load_dotenv()

# 필요한 모듈 가져오기
from utils.web_search import MedicalWebSearch
from utils.web_scraper import MedicalWebScraper
from rag.web_retriever import WebRetriever
from agent import MedicalAgent

# 테스트 쿼리
TEST_QUERIES = [
    '당뇨병 최신 치료법',
    'COVID-19 후유증 관리',
    '고혈압 가이드라인',
    '소아 비만 예방',
    '불면증 치료 방법'
]

async def test_web_search():
    """웹 검색 기능 테스트"""
    
    print('=== 웹 검색 기능 테스트 ===')
    
    # MedicalWebSearch 직접 사용 테스트
    try:
        searcher = MedicalWebSearch(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            google_cse_id=os.getenv('GOOGLE_CSE_ID')
        )
        
        query = TEST_QUERIES[0]
        print(f'\n[MedicalWebSearch] 쿼리: \'{query}\'')
        
        # 일반 검색
        results = searcher.search(query, num_results=3)
        print(f'일반 검색 결과: {len(results)}개')
        for i, result in enumerate(results, 1):
            print(f'  {i}. {result[\'title\']}')
            print(f'     URL: {result[\'link\']}')
            print(f'     발췌: {result[\'snippet\'][:100]}...')
        
        # 의학 정보 특화 검색
        medical_results = searcher.search_medical_info(query, num_results=3)
        print(f'\n의학 정보 특화 검색 결과: {len(medical_results)}개')
        for i, result in enumerate(medical_results, 1):
            print(f'  {i}. {result[\'title\']}')
            print(f'     URL: {result[\'link\']}')
            print(f'     발췌: {result[\'snippet\'][:100]}...')
        
        # 의학 저널 특화 검색
        journal_results = searcher.search_medical_journals(query, num_results=3)
        print(f'\n의학 저널 특화 검색 결과: {len(journal_results)}개')
        for i, result in enumerate(journal_results, 1):
            print(f'  {i}. {result[\'title\']}')
            print(f'     URL: {result[\'link\']}')
            print(f'     발췌: {result[\'snippet\'][:100]}...')
            
    except Exception as e:
        logger.error(f'MedicalWebSearch 테스트 실패: {e}', exc_info=True)
        return False
    
    return True

async def test_web_scraper():
    """웹 스크래퍼 기능 테스트"""
    
    print('\n=== 웹 스크래퍼 기능 테스트 ===')
    
    try:
        # MedicalWebScraper 직접 사용 테스트
        scraper = MedicalWebScraper()
        
        # 샘플 URL (테스트용)
        # PubMed 아티클
        pubmed_url = 'https://pubmed.ncbi.nlm.nih.gov/34668441/'
        
        print(f'\n[MedicalWebScraper] URL: {pubmed_url}')
        result = scraper.scrape_medical_journal(pubmed_url)
        
        if 'error' in result:
            print(f'  스크래핑 실패: {result[\'error\']}')
        else:
            print(f'  제목: {result.get(\'title\')}')
            print(f'  저자: {\', \'.join(result.get(\'authors\', []))}')
            print(f'  초록: {result.get(\'abstract\', \'\')[:150]}...')
            print(f'  DOI: {result.get(\'doi\')}')
            print(f'  저널: {result.get(\'journal\')}')
        
    except Exception as e:
        logger.error(f'MedicalWebScraper 테스트 실패: {e}', exc_info=True)
        return False
    
    return True

async def test_web_retriever():
    """WebRetriever 기능 테스트"""
    
    print('\n=== WebRetriever 기능 테스트 ===')
    
    try:
        # WebRetriever 직접 사용 테스트
        retriever = WebRetriever()
        
        query = TEST_QUERIES[1]
        print(f'\n[WebRetriever] 쿼리: \'{query}\'')
        
        documents = await retriever.retrieve_documents(query, limit=3)
        print(f'검색된 문서: {len(documents)}개')
        
        for i, doc in enumerate(documents, 1):
            print(f'  {i}. {doc.title}')
            print(f'     ID: {doc.id}')
            print(f'     내용: {doc.content[:100]}...')
            print(f'     소스: {doc.metadata.get(\'source_url\')}')
            print(f'     유형: {doc.metadata.get(\'type\')}')
            
    except Exception as e:
        logger.error(f'WebRetriever 테스트 실패: {e}', exc_info=True)
        return False
    
    return True

async def test_agent_integration():
    """에이전트 통합 테스트"""
    
    print('\n=== 에이전트 통합 테스트 ===')
    
    try:
        # MedicalAgent 생성
        agent = MedicalAgent()
        
        # 웹 검색 기능 활성화 상태 확인
        if not agent.web_search_enabled:
            print('웹 검색 기능이 비활성화되어 있습니다. .env 파일의 ENABLE_WEB_SEARCH 설정을 확인하세요.')
            return False
        
        query = TEST_QUERIES[2]
        print(f'\n[MedicalAgent] 쿼리: \'{query}\'')
        
        # 직접 웹 검색 수행
        search_results = await agent.direct_web_search(query, search_type='medical', num_results=3)
        print(f'직접 웹 검색 결과: {len(search_results)}개')
        
        # 전체 프로세스 테스트
        result = agent.process_query(query)
        
        print('\n[에이전트 응답]')
        print(f'{result[\'response\']}')
        
        print('\n[소스 정보]')
        for i, source in enumerate(result['sources'], 1):
            print(f'  {i}. {source[\'title\']}')
            if 'metadata' in source and 'source_url' in source['metadata']:
                print(f'     URL: {source[\'metadata\'][\'source_url\']}')
        
        print('\n[메타데이터]')
        print(f'  처리 시간: {result[\'metadata\'].get(\'processing_time\', 0):.2f}초')
        print(f'  웹 검색 사용: {result[\'metadata\'].get(\'used_web_search\', False)}')
        
    except Exception as e:
        logger.error(f'에이전트 통합 테스트 실패: {e}', exc_info=True)
        return False
    
    return True

async def main():
    """테스트 실행"""
    
    # 환경 변수 검증
    if not os.getenv('ENABLE_WEB_SEARCH', '').lower() in ['true', '1', 'yes']:
        print('웹 검색 기능이 비활성화되어 있습니다. .env 파일에 ENABLE_WEB_SEARCH=true를 설정하세요.')
        return
    
    if not os.getenv('GOOGLE_API_KEY'):
        print('GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일에서 설정하세요.')
        return
    
    if not os.getenv('GOOGLE_CSE_ID'):
        print('GOOGLE_CSE_ID가 설정되지 않았습니다. .env 파일에서 설정하세요.')
        return
    
    # 테스트 실행
    web_search_success = await test_web_search()
    web_scraper_success = await test_web_scraper()
    web_retriever_success = await test_web_retriever()
    agent_integration_success = await test_agent_integration()
    
    # 결과 요약
    print('\n=== 테스트 결과 요약 ===')
    print(f'웹 검색 테스트: {\'성공\' if web_search_success else \'실패\'}')
    print(f'웹 스크래퍼 테스트: {\'성공\' if web_scraper_success else \'실패\'}')
    print(f'웹 리트리버 테스트: {\'성공\' if web_retriever_success else \'실패\'}')
    print(f'에이전트 통합 테스트: {\'성공\' if agent_integration_success else \'실패\'}')
    
    if all([web_search_success, web_scraper_success, web_retriever_success, agent_integration_success]):
        print('\n모든 테스트가 성공적으로 완료되었습니다!')
    else:
        print('\n일부 테스트가 실패했습니다. 로그를 확인하세요.')

if __name__ == '__main__':
    asyncio.run(main()) 
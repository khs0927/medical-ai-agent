#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Custom Search Engine 생성 및 설정 스크립트
"""

import os
import json
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 키 설정
API_KEY = os.getenv('GOOGLE_API_KEY')

# 의학 관련 사이트 목록
MEDICAL_SITES = [
    'pubmed.ncbi.nlm.nih.gov',
    'nejm.org',
    'who.int',
    'mayoclinic.org',
    'jamanetwork.com',
    'thelancet.com',
    'bmj.com',
    'medscape.com',
    'webmd.com',
    'cdc.gov',
    'nih.gov',
    'healthline.com',
    'uptodate.com'
]

def create_custom_search_engine():
    """Custom Search Engine 생성"""
    if not API_KEY:
        print("Google API 키가 설정되지 않았습니다.")
        print("API 키를 .env 파일에 다음과 같이 설정해주세요: GOOGLE_API_KEY=your_api_key")
        return False
        
    try:
        # CustomSearchAPI 서비스 초기화
        service = googleapiclient.discovery.build(
            'customsearch', 'v1', developerKey=API_KEY
        )
        
        print('Google Custom Search API에 연결했습니다.')
        print('API 키가 환경변수에서 로드되었습니다.')
        
        # 서비스가 정상적으로 작동하는지 테스트
        search_results = service.cse().list(
            q='medical research',
            cx='017576662512468239146:omuauf_lfve',  # 테스트용 공개 CSE ID
        ).execute()
        
        print('API 연결 테스트 성공!')
        
        print('\n=== 중요 안내사항 ===')
        print('Google Custom Search Engine ID(CSE ID)를 생성하려면 다음 단계를 따르세요:')
        print('1. https://programmablesearchengine.google.com/에 접속하세요')
        print('2. \'새 검색 엔진\'을 클릭하세요')
        print('3. 다음 의학 관련 사이트를 추가하세요:')
        for site in MEDICAL_SITES:
            print(f'   - {site}')
        print('4. \'만들기\' 버튼을 클릭하세요')
        print('5. 생성 후 \'수정\' 버튼을 클릭하고 \'검색 엔진 ID\'를 찾아 복사하세요')
        print('6. 이 CSE ID를 .env 파일에 추가하세요:')
        print('   GOOGLE_CSE_ID=\'복사한_CSE_ID\'')
        
        return True
    
    except HttpError as error:
        print(f'API 호출 에러: {error}')
        return False
    
    except Exception as e:
        print(f'오류 발생: {e}')
        return False

def update_env_file(cse_id=None):
    """환경 변수 파일 업데이트"""
    try:
        env_content = 'ENABLE_WEB_SEARCH=true\n'
        env_content += '# API 키는 별도로 설정해야 합니다\n'
        env_content += '# GOOGLE_API_KEY=your_google_api_key\n'
        
        if cse_id:
            env_content += f'GOOGLE_CSE_ID={cse_id}\n'
        else:
            env_content += '# GOOGLE_CSE_ID를 아래에 추가하세요\n'
            env_content += '# GOOGLE_CSE_ID=your_custom_search_engine_id\n'
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print('.env 파일을 성공적으로 업데이트했습니다.')
        print('주의: API 키는 반드시 직접 .env 파일에 추가해야 합니다.')
        return True
    
    except Exception as e:
        print(f'.env 파일 업데이트 중 오류 발생: {e}')
        return False

if __name__ == '__main__':
    print('=== Google Custom Search Engine 설정 도우미 ===')
    create_custom_search_engine()
    update_env_file() 
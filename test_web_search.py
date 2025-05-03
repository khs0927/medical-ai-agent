#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
의료 에이전트의 웹 검색 기능을 테스트하는 스크립트
"""

import os
import json
import dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def load_env():
    """환경 변수 로드"""
    dotenv.load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    
    return api_key, cse_id

def test_google_search(api_key, cse_id, query='당뇨병 최신 치료법'):
    """Google 검색 테스트"""
    try:
        # API 연결
        service = build('customsearch', 'v1', developerKey=api_key)
        
        # 검색 요청
        res = service.cse().list(
            q=query,
            cx=cse_id,
            num=3  # 결과 3개만 가져오기
        ).execute()
        
        # 결과 출력
        if 'items' in res:
            print(f'\n\'{query}\'에 대한 검색 결과:')
            for i, item in enumerate(res['items'], 1):
                print(f'\n--- 결과 {i} ---')
                print(f'제목: {item[\'title\']}')
                print(f'링크: {item[\'link\']}')
                print(f'스니펫: {item.get(\'snippet\', \'(설명 없음)\')[:100]}...')
            
            print('\n✅ 검색 테스트가 성공적으로 완료되었습니다!')
            return True
        else:
            print(f'검색 결과가 없습니다: {query}')
            return False
            
    except HttpError as e:
        error_details = json.loads(e.content.decode())
        error_reason = error_details.get('error', {}).get('errors', [{}])[0].get('reason', 'unknown')
        
        print(f'\n⚠️ API 호출 오류: {e}')
        
        if error_reason == 'keyInvalid':
            print('API 키가 유효하지 않습니다. .env 파일에서 GOOGLE_API_KEY를 확인하세요.')
        elif error_reason == 'invalid':
            print('검색 엔진 ID(CSE ID)가 유효하지 않습니다. .env 파일에서 GOOGLE_CSE_ID를 확인하세요.')
        elif error_reason == 'dailyLimitExceeded':
            print('일일 API 호출 한도를 초과했습니다.')
        else:
            print(f'오류 원인: {error_reason}')
            
        return False
        
    except Exception as e:
        print(f'\n⚠️ 오류 발생: {e}')
        return False

def main():
    """메인 함수"""
    print('=== 의료 에이전트 웹 검색 테스트 ===')
    
    # 환경 변수 로드
    api_key, cse_id = load_env()
    
    # 환경 변수 확인
    if not api_key:
        print('⚠️ API 키가 설정되지 않았습니다. .env 파일에 GOOGLE_API_KEY를 추가하세요.')
        return False
    
    if not cse_id:
        print('⚠️ 검색 엔진 ID가 설정되지 않았습니다. .env 파일에 GOOGLE_CSE_ID를 추가하세요.')
        print('CSE ID를 설정하려면 add_cse_id.py 스크립트를 실행하세요.')
        return False
    
    print(f'API 키: {api_key[:5]}...{api_key[-5:]}')
    print(f'CSE ID: {cse_id}')
    
    # 검색 테스트
    return test_google_search(api_key, cse_id)

if __name__ == '__main__':
    main() 
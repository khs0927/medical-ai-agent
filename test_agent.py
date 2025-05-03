#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
의료 AI 에이전트 테스트 스크립트
테스트 모드에서 에이전트를 실행하여 기능을 검증합니다.
"""
import os
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv
import requests

# 환경 변수 로드
load_dotenv('.env.test')

# 테스트 모드 설정
os.environ['TEST_MODE'] = 'true'

# 테스트할 질문 목록
TEST_QUESTIONS = [
    '아스피린과 클로피도그렐 동시 복용해도 되나요?',
    '심근경색 환자 응급 처치 가이드라인을 알려주세요.',
    '최근에 가슴 통증과 호흡 곤란이 있어요. 위험한가요?',
    '고혈압 환자의 식이요법에 대해 알려주세요.',
]

def start_server():
    """FastAPI 서버를 시작합니다."""
    print('서버를 시작합니다...')
    # 백그라운드에서 서버 실행
    server_process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'src.medical_agent.__main__:app', '--reload', '--port', '8000'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if os.name != 'nt' else None,
    )

    # 서버가 시작될 때까지 대기
    time.sleep(5)
    print('서버가 시작되었습니다.')
    return server_process

def stop_server(server_process):
    """서버를 중지합니다."""
    print('서버를 종료합니다...')
    if os.name == 'nt':
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(server_process.pid)])
    else:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    print('서버가 종료되었습니다.')

def test_api():
    """API 엔드포인트를 테스트합니다."""
    base_url = 'http://localhost:8000'

    # 헬스 체크
    try:
        response = requests.get(f'{base_url}/healthz')
        print(f'Health Check: {response.status_code} - {response.json()}')
    except Exception as e:
        print(f'Health Check 오류: {e}')
        return False

    # 상담 API 테스트
    success = True
    for i, question in enumerate(TEST_QUESTIONS, 1):
        try:
            print(f'\n테스트 {i}: \'{question}\'')
            response = requests.post(
                f'{base_url}/v1/consult',
                json={'question': question},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                print(f'상태: 성공 (200 OK)')
                print(f'응답:\n{result[\'answer\']}\n')
                print('-' * 80)
            else:
                print(f'상태: 실패 ({response.status_code})')
                print(f'오류: {response.text}')
                success = False
        except Exception as e:
            print(f'API 호출 오류: {e}')
            success = False

    return success

def main():
    """메인 테스트 함수"""
    print('=' * 80)
    print('의료 AI 에이전트 테스트 시작')
    print('=' * 80)

    server_process = None
    try:
        server_process = start_server()

        # API 테스트
        success = test_api()

        if success:
            print('\n✅ 모든 테스트가 성공적으로 완료되었습니다.')
        else:
            print('\n❌ 일부 테스트에 실패했습니다.')

    finally:
        if server_process:
            stop_server(server_process)

    print('\n테스트 종료')

if __name__ == '__main__':
    main()

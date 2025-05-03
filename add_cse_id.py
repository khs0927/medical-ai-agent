#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Custom Search Engine ID를 .env 파일에 추가하는 스크립트
"""

import os

def update_env_with_cse_id():
    """사용자로부터 CSE ID를 입력받아 .env 파일에 추가"""
    print('=== Google Custom Search Engine ID 설정 ===')
    print('프로그래밍 가능한 검색 엔진(https://programmablesearchengine.google.com/)에서')
    print('생성한 CSE ID를 입력해주세요.')
    
    cse_id = input('CSE ID: ').strip()
    
    if not cse_id:
        print('CSE ID가 입력되지 않았습니다. 설정을 취소합니다.')
        return False
    
    try:
        # 기존 .env 파일 읽기
        env_content = ''
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_content = f.read()
        
        # CSE_ID가 이미 있는지 확인
        if 'GOOGLE_CSE_ID' in env_content:
            # 기존 CSE_ID 라인 교체
            lines = env_content.splitlines()
            new_lines = []
            
            for line in lines:
                if line.startswith('GOOGLE_CSE_ID=') or line.startswith('# GOOGLE_CSE_ID='):
                    new_lines.append(f'GOOGLE_CSE_ID={cse_id}')
                else:
                    new_lines.append(line)
            
            env_content = '\n'.join(new_lines)
        else:
            # CSE_ID 추가
            if env_content and not env_content.endswith('\n'):
                env_content += '\n'
            env_content += f'GOOGLE_CSE_ID={cse_id}\n'
        
        # 파일에 저장
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print(f'CSE ID \'{cse_id}\'를 .env 파일에 성공적으로 저장했습니다.')
        print('이제 웹 검색 기능을 사용할 수 있습니다.')
        return True
    
    except Exception as e:
        print(f'오류 발생: {e}')
        return False

if __name__ == '__main__':
    update_env_with_cse_id() 
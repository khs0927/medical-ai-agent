#!/usr/bin/env python3
"""
코드 문제 자동 분석 및 수정 스크립트.
MCP가 없어도 실행 가능한 버전입니다.
"""

# 로깅 설정
import logging
import os
import re
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('code-fixer')

# 분석 대상 디렉토리 및 파일 확장자 설정
TARGET_DIRS = [
    '.',
    'src/',
    'components/',
    'data/',
    'db/',
    'rag/',
    'fastapi_app/',
    'models/'
]

# 분석할 파일 확장자
FILE_EXTENSIONS = ['.py', '.js', '.ts', '.tsx', '.html', '.css', '.json', '.yml', '.yaml']

# 가장 많은 문제가 발견된 파일 목록
PROBLEM_FILES = [
    'agent.py', 
    'deploy_to_github.py', 
    'run_server.py', 
    'run_server_production.py', 
    'test_agent.py',
    'client.py',
    'client_production.py'
]

def install_dependencies() -> None:
    """
    필요한 종속성을 설치합니다.
    """
    logger.info('필요한 패키지를 설치합니다...')
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pylint', 'black', 'isort', 'autoflake'], check=True)
        logger.info('패키지 설치가 완료되었습니다.')
    except subprocess.CalledProcessError as e:
        logger.error(f'패키지 설치 중 오류가 발생했습니다: {e}')
        sys.exit(1)

def fix_indentation_issues() -> None:
    """
    들여쓰기 문제를 해결합니다. (특히 agent.py, deploy_to_github.py 등)
    """
    logger.info('들여쓰기 문제 해결 중...')
    
    for file_name in PROBLEM_FILES:
        if not os.path.exists(file_name):
            continue
            
        try:
            # 파일 내용 읽기
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 들여쓰기 수정 (잘못된 들여쓰기를 4칸 스페이스로 변경)
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                if line.strip():  # 빈 줄이 아닌 경우
                    # 현재 들여쓰기 수준 계산
                    indent_level = len(line) - len(line.lstrip())
                    if indent_level % 4 != 0:
                        # 올바른 들여쓰기 레벨로 조정 (4의 배수로)
                        correct_level = (indent_level // 4) * 4
                        fixed_line = ' ' * correct_level + line.lstrip()
                    else:
                        fixed_line = line
                else:
                    fixed_line = line
                
                # 줄 끝 공백 제거
                fixed_line = fixed_line.rstrip()
                fixed_lines.append(fixed_line)
            
            # 마지막 줄 개행 추가
            if fixed_lines and fixed_lines[-1]:
                fixed_lines.append('')
            
            # 수정된 내용 저장
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
                
            logger.info(f'\'{file_name}\' 들여쓰기 수정 완료')
        except Exception as e:
            logger.error(f'\'{file_name}\' 들여쓰기 수정 중 오류 발생: {e}')

def fix_logging_format() -> None:
    """
    로깅 포맷 문제(f-string → lazy %)를 해결합니다.
    """
    problem_files = [
        'agent.py', 
        'deploy_to_github.py'
    ]
    
    logger.info('로깅 포맷 문제 해결 중...')
    
    for file_name in problem_files:
        if not os.path.exists(file_name):
            continue
            
        try:
            # 파일 내용 읽기
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # f-string 로깅을 lazy % 포맷으로 변경
            # logger.xxx(f'text {var}') → logger.xxx('text %s', var)
            pattern = r'(logger\.\w+)\s*\(\s*f[\''](.+?)[\\''](\s*\))'
            
            def replace_logging(match):
                logger_call = match.group(1)
                msg = match.group(2)
                closing = match.group(3)
                
                # 변수 추출
                vars = re.findall(r'\{(.+?)\}', msg)
                
                # 포맷 문자열로 변환
                formatted_msg = re.sub(r'\{(.+?)\}', '%s', msg)
                
                # 변수 파라미터 추가
                if vars:
                    vars_str = ', ' + ', '.join(vars)
                    return f'{logger_call}('{formatted_msg}'{vars_str}{closing}'
                else:
                    return f'{logger_call}('{formatted_msg}'{closing}'
            
            modified_content = re.sub(pattern, replace_logging, content)
            
            # 수정된 내용 저장
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(modified_content)
                
            logger.info(f'\'{file_name}\' 로깅 포맷 수정 완료')
        except Exception as e:
            logger.error(f'\'{file_name}\' 로깅 포맷 수정 중 오류 발생: {e}')

def format_code_with_black_and_isort() -> None:
    """
    Black과 isort를 사용하여 Python 코드를 자동으로 포맷팅합니다.
    """
    logger.info('Black 및 isort로 코드 포맷팅 중...')
    
    for target_dir in TARGET_DIRS:
        if not os.path.exists(target_dir):
            continue
            
        try:
            # Black으로 코드 포맷팅
            subprocess.run([sys.executable, '-m', 'black', target_dir], check=False)
            
            # isort로 import 정리
            subprocess.run([sys.executable, '-m', 'isort', target_dir], check=False)
            
            logger.info(f'\'{target_dir}\' 포맷팅 완료')
        except Exception as e:
            logger.error(f'\'{target_dir}\' 포맷팅 중 오류 발생: {e}')

def remove_unused_imports() -> None:
    """
    사용하지 않는 import 문을 제거합니다.
    """
    logger.info('사용하지 않는 import 제거 중...')
    
    for target_dir in TARGET_DIRS:
        if not os.path.exists(target_dir):
            continue
            
        try:
            # autoflake로 사용하지 않는 import 제거
            subprocess.run([
                sys.executable, 
                '-m', 
                'autoflake', 
                '--remove-all-unused-imports', 
                '--recursive', 
                '--in-place', 
                target_dir
            ], check=False)
            
            logger.info(f'\'{target_dir}\' 의 사용하지 않는 import 제거 완료')
        except Exception as e:
            logger.error(f'\'{target_dir}\' 의 import 제거 중 오류 발생: {e}')

def run_pylint() -> None:
    """
    pylint를 실행하여 코드 문제를 확인합니다.
    """
    logger.info('Pylint로 코드 문제 확인 중...')
    
    try:
        # pylint 결과 저장 파일
        output_file = 'pylint_results.txt'
        
        # 모든 Python 파일 찾기
        python_files = []
        for target_dir in TARGET_DIRS:
            if os.path.exists(target_dir):
                if target_dir == '.':
                    # 최상위 디렉토리의 Python 파일만 추가
                    python_files.extend([f for f in os.listdir('.') if f.endswith('.py')])
                else:
                    # 하위 디렉토리 포함
                    for root, _, files in os.walk(target_dir):
                        python_files.extend([os.path.join(root, f) for f in files if f.endswith('.py')])
        
        # 중복 제거 및 정렬
        python_files = sorted(set(python_files))
        
        if python_files:
            # pylint 실행
            with open(output_file, 'w', encoding='utf-8') as f:
                subprocess.run([sys.executable, '-m', 'pylint'] + python_files, stdout=f, stderr=subprocess.STDOUT, check=False)
            
            logger.info(f'Pylint 결과가 \'{output_file}\'에 저장되었습니다.')
        else:
            logger.warning('분석할 Python 파일을 찾을 수 없습니다.')
            
    except Exception as e:
        logger.error(f'Pylint 실행 중 오류 발생: {e}')

def main() -> None:
    """
    메인 함수입니다.
    """
    logger.info('코드 자동 수정 시작')
    
    try:
        # 1. 필요한 의존성 설치
        install_dependencies()
        
        # 2. 들여쓰기 문제 해결
        fix_indentation_issues()
        
        # 3. 로깅 포맷 수정
        fix_logging_format()
        
        # 4. Black과 isort로 코드 포맷팅
        format_code_with_black_and_isort()
        
        # 5. 사용하지 않는 import 제거
        remove_unused_imports()
        
        # 6. 최종 코드 분석 실행
        run_pylint()
        
        logger.info('모든 자동 수정이 완료되었습니다.')
        print('\n===== 주요 수정 사항 =====')
        print('1. 들여쓰기 문제 해결 (4칸 스페이스로 통일)')
        print('2. 줄 끝 공백 제거 및 마지막 줄 개행 추가')
        print('3. 로깅 f-string을 lazy % 포맷으로 변경')
        print('4. Black과 isort로 코드 자동 포맷팅')
        print('5. 사용하지 않는 import 제거')
        print('6. pylint 결과 분석 보고서 생성 (pylint_results.txt)')
        print('\n남은 문제를 확인하려면 \'pylint_results.txt\' 파일을 참조하세요.')
        
    except Exception as e:
        logger.error(f'오류 발생: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
들여쓰기 문제를 해결하는 전용 스크립트.
특히 agent.py 파일에 집중하여 파이썬 코드 들여쓰기를 수정합니다.
"""

import logging
import os
import sys
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('indentation-fixer')

# 가장 많은 문제가 발견된 파일 목록
PROBLEM_FILES = [
    'agent.py', 
    'deploy_to_github.py', 
    'src/medical_agent/tools.py',
    'src/medical_agent/retriever.py',
    'src/medical_agent/agents.py',
    'src/medical_agent/schemas.py',
    'src/medical_agent/hf_client.py'
]

def fix_python_indentation(file_path, expected_indent=4):
    """
    파이썬 파일의 들여쓰기를 수정합니다.
    
    Args:
        file_path: 수정할 파일 경로
        expected_indent: 기대하는 들여쓰기 간격 (기본값: 4)
    """
    if not os.path.exists(file_path):
        logger.warning(f'파일을 찾을 수 없음: {file_path}')
        return
    
    try:
        # 파일 내용 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 들여쓰기 수준 추적 변수
        current_indent_level = 0
        fixed_lines = []
        
        # 클래스, 함수 정의 등의 들여쓰기 변화를 감지하는 키워드
        indent_increase_keywords = ['class ', 'def ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except:',
                                   'else:', 'elif ', 'finally:']
        
        for line in lines:
            # 빈 줄이나 주석만 있는 줄은 그대로 유지
            if not line.strip() or line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
            
            # 현재 줄의 들여쓰기 수준 계산
            current_line_indent = len(line) - len(line.lstrip())
            clean_line = line.strip()
            
            # 들여쓰기 수준 계산 (코드 구조를 분석)
            for keyword in indent_increase_keywords:
                if clean_line.startswith(keyword) and clean_line.endswith(':'):
                    # 다음 줄부터 들여쓰기 수준 증가
                    current_indent_level += 1
                    break
            
            # 닫는 괄호나 들여쓰기 감소를 감지
            if clean_line == '}' or clean_line == ']' or clean_line == ')' or clean_line == 'else:' or \
               clean_line == 'elif:' or clean_line == 'except:' or clean_line == 'finally:':
                current_indent_level = max(0, current_indent_level - 1)
            
            # 올바른 들여쓰기 계산
            proper_indent = current_indent_level * expected_indent
            fixed_line = ' ' * proper_indent + line.lstrip()
            fixed_lines.append(fixed_line)
        
        # 수정된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        logger.info(f'들여쓰기 수정 완료: {file_path}')
    
    except Exception as e:
        logger.error(f'들여쓰기 수정 중 오류 발생 ({file_path}): {e}')

def fix_indentation_with_black(file_path):
    """
    Black을 사용하여 파이썬 파일의 들여쓰기를 수정합니다.
    
    Args:
        file_path: 수정할 파일 경로
    """
    if not os.path.exists(file_path):
        logger.warning(f'파일을 찾을 수 없음: {file_path}')
        return
    
    try:
        import black
        
        logger.info(f'Black으로 포맷팅 중: {file_path}')
        
        # Black의 파일 모드를 사용하여 파일 포맷팅
        black.format_file_in_place(
            Path(file_path), 
            fast=False, 
            mode=black.FileMode(line_length=88)
        )
        
        logger.info(f'Black 포맷팅 완료: {file_path}')
    
    except ImportError:
        logger.warning('Black 패키지가 설치되어 있지 않습니다. pip install black으로 설치하세요.')
    
    except Exception as e:
        logger.error(f'Black 포맷팅 중 오류 발생 ({file_path}): {e}')

def simple_indentation_fix(file_path, expected_indent=4):
    """
    간단한 규칙으로 들여쓰기를 수정합니다.
    
    Args:
        file_path: 수정할 파일 경로
        expected_indent: 기대하는 들여쓰기 간격 (기본값: 4)
    """
    if not os.path.exists(file_path):
        logger.warning(f'파일을 찾을 수 없음: {file_path}')
        return
    
    try:
        # 파일 내용 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        
        for line in lines:
            # 빈 줄은 그대로 유지
            if not line.strip():
                fixed_lines.append(line)
                continue
            
            # 현재 들여쓰기 수준 계산
            current_indent = len(line) - len(line.lstrip())
            
            # 들여쓰기 수준이 expected_indent의 배수가 아니면 수정
            if current_indent % expected_indent != 0:
                # 가장 가까운 expected_indent의 배수로 조정
                proper_indent = (current_indent // expected_indent) * expected_indent
                fixed_line = ' ' * proper_indent + line.lstrip()
            else:
                fixed_line = line
            
            fixed_lines.append(fixed_line)
        
        # 수정된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        logger.info(f'간단한 들여쓰기 수정 완료: {file_path}')
    
    except Exception as e:
        logger.error(f'들여쓰기 수정 중 오류 발생 ({file_path}): {e}')

def remove_trailing_whitespace(file_path):
    """
    줄 끝의 공백을 제거하고 파일 끝에 개행을 추가합니다.
    
    Args:
        file_path: 수정할 파일 경로
    """
    if not os.path.exists(file_path):
        logger.warning(f'파일을 찾을 수 없음: {file_path}')
        return
    
    try:
        # 파일 내용 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 줄 끝 공백 제거
        fixed_lines = [line.rstrip() + '\n' for line in lines]
        
        # 파일 끝 개행 확인 및 추가
        if fixed_lines and not fixed_lines[-1].endswith('\n'):
            fixed_lines[-1] += '\n'
        
        # 수정된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        logger.info(f'줄 끝 공백 제거 완료: {file_path}')
    
    except Exception as e:
        logger.error(f'줄 끝 공백 제거 중 오류 발생 ({file_path}): {e}')

def main():
    """
    메인 실행 함수
    """
    logger.info('들여쓰기 수정 시작')
    
    try:
        for file_path in PROBLEM_FILES:
            if os.path.exists(file_path):
                logger.info(f'파일 처리 중: {file_path}')
                
                # 간단한 들여쓰기 수정 적용
                simple_indentation_fix(file_path)
                
                # 줄 끝 공백 제거 및 파일 끝 개행 추가
                remove_trailing_whitespace(file_path)
                
                logger.info(f'파일 처리 완료: {file_path}')
            else:
                logger.warning(f'파일을 찾을 수 없음: {file_path}')
        
        logger.info('모든 파일 처리 완료')
        
    except Exception as e:
        logger.error(f'들여쓰기 수정 중 오류 발생: {e}')
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
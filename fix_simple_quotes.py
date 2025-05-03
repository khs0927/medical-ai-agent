#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
간단한 따옴표 수정 스크립트

프로젝트 파일 내의 문자열 리터럴에서 큰따옴표(")를 작은따옴표(')로 일관되게 변경하는 스크립트입니다.
복잡한 AST 파싱 없이 기본 정규 표현식을 사용하여 단순하게 처리합니다.
"""

import os
import re
import logging
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 제외할 디렉토리 목록
EXCLUDE_DIRS = ['.venv', 'venv', 'env', '.git', '__pycache__', 'node_modules']

def find_py_files(root_dir):
    """주어진 디렉토리에서 모든 Python 파일을 찾습니다. 제외 디렉토리는 건너뜁니다."""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # 제외 디렉토리 제거
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def fix_quotes_in_file(file_path, files_to_process=None):
    """파일 내의 큰따옴표를 작은따옴표로 변경합니다."""
    # 특정 파일 목록이 제공되었는데 현재 파일이 목록에 없으면 처리하지 않음
    if files_to_process and os.path.basename(file_path) not in files_to_process:
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 문자열 내부의 따옴표는 건너뛰도록 처리
        # 1. r"..." 형태의 raw 문자열
        # 2. f"..." 형태의 f-string
        # 3. 일반 "..." 문자열
        
        # 패턴: r"..." -> r'...'
        r_string_pattern = r'r"((?:\\.|[^"\\])*)"'
        modified_content = re.sub(r_string_pattern, r"r'\1'", content)
        
        # 패턴: f"..." -> f'...'
        f_string_pattern = r'f"((?:\\.|[^"\\])*)"'
        modified_content = re.sub(f_string_pattern, r"f'\1'", modified_content)
        
        # 패턴: "..." -> '...'
        string_pattern = r'"((?:\\.|[^"\\])*)"'
        modified_content = re.sub(string_pattern, r"'\1'", modified_content)
        
        # 내용이 변경된 경우에만 파일을 다시 씀
        if content != modified_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            logger.info('따옴표 수정 완료: %s', file_path)
            return True
        else:
            logger.info('수정할 따옴표가 없습니다: %s', file_path)
            return False
        
    except Exception as e:
        logger.error('파일 %s 처리 중 오류 발생: %s', file_path, e)
        return False

def main():
    parser = argparse.ArgumentParser(description='큰따옴표를 작은따옴표로 일관되게 변경합니다.')
    parser.add_argument('--dir', type=str, default='.', help='처리할 디렉토리 (기본값: 현재 디렉토리)')
    parser.add_argument('--file', type=str, help='특정 파일만 처리 (선택 사항)')
    parser.add_argument('--exclude', type=str, nargs='+', help='추가로 제외할 디렉토리 지정')
    parser.add_argument('--files', type=str, nargs='+', help='처리할 파일 이름 목록')
    
    args = parser.parse_args()
    
    # 추가 제외 디렉토리 처리
    if args.exclude:
        for exclude_dir in args.exclude:
            if exclude_dir not in EXCLUDE_DIRS:
                EXCLUDE_DIRS.append(exclude_dir)
    
    if args.file:
        logger.info('파일 처리 중: %s', args.file)
        fix_quotes_in_file(args.file)
    else:
        logger.info('디렉토리 내 Python 파일 처리 중: %s', args.dir)
        logger.info('제외 디렉토리: %s', ', '.join(EXCLUDE_DIRS))
        
        py_files = find_py_files(args.dir)
        logger.info('발견된 Python 파일 수: %s', len(py_files))
        
        success_count = 0
        for file_path in py_files:
            if fix_quotes_in_file(file_path, args.files):
                success_count += 1
        
        logger.info('총 %s개 파일 중 %s개 파일 처리 완료', len(py_files), success_count)

if __name__ == '__main__':
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
따옴표 일관성 문제를 해결하기 위한 스크립트

이 스크립트는 프로젝트 내의 모든 파이썬 파일에서 따옴표를 단일 따옴표(')로 통일합니다.
일관성 문제(W1405) 해결을 위해 사용됩니다.
"""

import os
import re
import logging
import argparse
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_py_files(root_dir):
    """주어진 디렉토리에서 모든 Python 파일을 찾습니다."""
    py_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def fix_quotes_in_file(file_path):
    """파일 내의 모든 따옴표를 단일 따옴표(')로 변경합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # docstring과 multiline string 처리
        # 먼저 모든 docstring과 multiline string을 찾아서 마커로 치환
        multiline_pattern = r'"""[\s\S]*?"""'
        
        # 문자열 내에서의 따옴표 이스케이핑 처리
        # docstring과 multiline string에서 먼저 찾아서 보존
        docstrings = re.findall(multiline_pattern, content, re.DOTALL)
        markers = [f'__DOCSTRING_MARKER_{i}__' for i in range(len(docstrings))]
        
        # 마커로 치환
        for i, docstring in enumerate(docstrings):
            content = content.replace(docstring, markers[i])
        
        # 일반 문자열의 따옴표 변경
        # 이중 따옴표로 된 문자열을 단일 따옴표로 변경
        # 이스케이프된 따옴표 처리
        double_quoted_pattern = r''([^'\\]*(?:\\.[^'\\]*)*)''
        
        # 정규식으로 이중 따옴표 문자열을 찾아 단일 따옴표로 변경
        for match in re.finditer(double_quoted_pattern, content):
            old_str = match.group(0)
            # 이중 따옴표 안의 내용에서 단일 따옴표를 이스케이프
            inner_content = match.group(1)
            inner_content = inner_content.replace('\'', '\\\'')
            # 이중 따옴표를 단일 따옴표로 변경
            new_str = f'\'{inner_content}\''
            content = content.replace(old_str, new_str)
        
        # docstring과 multiline string을 원래대로 복원
        for i, docstring in enumerate(docstrings):
            # Docstring은 단일 따옴표로 변경하지 않음
            content = content.replace(markers[i], docstring)
        
        # 변경된 내용을 파일에 쓰기
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f'따옴표 수정 완료: {file_path}')
        return True
    
    except Exception as e:
        logger.error(f'파일 {file_path} 처리 중 오류 발생: {e}')
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Python 파일의 따옴표를 단일 따옴표로 통일합니다.')
    parser.add_argument('--dir', type=str, default='.', help='처리할 디렉토리 (기본값: 현재 디렉토리)')
    parser.add_argument('--file', type=str, help='특정 파일만 처리 (선택 사항)')
    
    args = parser.parse_args()
    
    if args.file:
        logger.info(f'파일 처리 중: {args.file}')
        fix_quotes_in_file(args.file)
    else:
        logger.info(f'디렉토리 내 모든 Python 파일 처리 중: {args.dir}')
        py_files = find_py_files(args.dir)
        logger.info(f'발견된 Python 파일 수: {len(py_files)}')
        
        success_count = 0
        for file_path in py_files:
            if fix_quotes_in_file(file_path):
                success_count += 1
        
        logger.info(f'총 {len(py_files)}개 파일 중 {success_count}개 파일 처리 완료')

if __name__ == '__main__':
    main() 
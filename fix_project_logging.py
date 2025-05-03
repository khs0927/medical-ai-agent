#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
프로젝트 파일만 대상으로 로깅 f-string을 수정하는 스크립트

이 스크립트는 프로젝트 디렉토리 내의 파일만 처리하여 로깅에서 f-string 대신 
lazy % 형식을 사용하도록 변환합니다.
예: logger.info(f'값: {value}') -> logger.info('값: %s', value)
"""

import os
import re
import ast
import logging
import argparse
from typing import List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 로깅 함수 목록
LOGGING_FUNCTIONS = [
    'debug', 'info', 'warning', 'error', 'critical', 
    'exception', 'log'
]

# 제외할 디렉토리 목록
EXCLUDE_DIRS = ['.venv', 'venv', 'env', '.git', '__pycache__', 'node_modules']

def find_py_files(root_dir: str) -> List[str]:
    """주어진 디렉토리에서 모든 Python 파일을 찾습니다. 제외 디렉토리는 건너뜁니다."""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # 제외 디렉토리 제거
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

class LoggingVisitor(ast.NodeVisitor):
    """AST 방문자 클래스로 f-string 로깅을 찾아서 수정합니다."""
    
    def __init__(self):
        self.replacements = []
        
    def visit_Call(self, node):
        """메서드 호출 노드를 방문하여 로깅 f-string을 찾습니다."""
        # logger.info(f'...') 형태 확인
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr in LOGGING_FUNCTIONS and 
            node.args and 
            isinstance(node.args[0], ast.JoinedStr)):
            
            # 위치 정보 저장
            lineno = node.lineno
            col_offset = node.col_offset
            end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else lineno
            end_col_offset = node.end_col_offset if hasattr(node, 'end_col_offset') else 0
            
            # 원본 f-string 분석
            f_string = node.args[0]
            format_string, values = self._extract_from_f_string(f_string)
            
            # 치환할 정보 저장
            self.replacements.append({
                'start': (lineno, col_offset),
                'end': (end_lineno, end_col_offset),
                'format_string': format_string,
                'values': values,
                'original': node
            })
        
        # 모든 자식 노드 방문
        self.generic_visit(node)
    
    def _extract_from_f_string(self, f_string: ast.JoinedStr) -> Tuple[str, List[str]]:
        """f-string에서 형식 문자열과 값들을 추출합니다."""
        format_parts = []
        values = []
        
        for value in f_string.values:
            if isinstance(value, ast.Constant):
                # 일반 문자열 부분
                format_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                # 표현식 부분 ({value} 등)
                format_parts.append('%s')  # 기본적으로 %s 사용
                
                # 값 추출
                if isinstance(value.value, ast.Name):
                    values.append(value.value.id)
                elif isinstance(value.value, ast.Attribute):
                    # obj.attr 형태
                    attr_parts = []
                    node = value.value
                    while isinstance(node, ast.Attribute):
                        attr_parts.insert(0, node.attr)
                        node = node.value
                    if isinstance(node, ast.Name):
                        attr_parts.insert(0, node.id)
                    values.append('.'.join(attr_parts))
                else:
                    # 복잡한 표현식은 그대로 코드로 추출
                    values.append(ast.unparse(value.value))
        
        return ''.join(format_parts), values

def fix_logging_in_file(file_path: str) -> bool:
    """파일의 로깅 f-string을 lazy % 형식으로 변경합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # AST 분석
        tree = ast.parse(content)
        visitor = LoggingVisitor()
        visitor.visit(tree)
        
        if not visitor.replacements:
            logger.info('파일에 수정할 로깅 f-string이 없습니다: %s', file_path)
            return True
        
        # 라인별로 내용 분리
        lines = content.splitlines(True)  # 개행 문자 유지
        
        # 뒤에서부터 치환하여 인덱스 변화 방지
        for replacement in sorted(visitor.replacements, key=lambda x: x['start'], reverse=True):
            start_line, start_col = replacement['start']
            end_line, end_col = replacement['end']
            format_string = replacement['format_string']
            values = replacement['values']
            
            # 변경할 텍스트 범위 찾기
            if start_line == end_line:
                # 한 라인 내에서 변경
                line = lines[start_line - 1]
                prefix = line[:start_col]
                suffix = line[end_col:]
                
                # logger.xyz 부분 추출
                match = re.match(r'^(.*?logger\.[a-z]+)\(', prefix)
                if match:
                    logger_part = match.group(1)
                    # 새 로깅 문 생성
                    if values:
                        values_joined = ', '.join(values)
                        new_line = f"{logger_part}('{format_string}', {values_joined}){suffix}"
                    else:
                        new_line = f"{logger_part}('{format_string}'){suffix}"
                    lines[start_line - 1] = new_line
            else:
                # 여러 라인에 걸친 경우 (덜 일반적)
                logger.warning('여러 줄에 걸친 로깅 문을 건너뜁니다: %s 라인 %s', file_path, start_line)
        
        # 변경된 내용 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info('로깅 문 %s개 수정 완료: %s', len(visitor.replacements), file_path)
        return True
    
    except Exception as e:
        logger.error('파일 %s 처리 중 오류 발생: %s', file_path, e)
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='로깅 f-string을 lazy % 형식으로 변환합니다.')
    parser.add_argument('--dir', type=str, default='.', help='처리할 디렉토리 (기본값: 현재 디렉토리)')
    parser.add_argument('--file', type=str, help='특정 파일만 처리 (선택 사항)')
    parser.add_argument('--exclude', type=str, nargs='+', help='추가로 제외할 디렉토리 지정')
    
    args = parser.parse_args()
    
    # 추가 제외 디렉토리 처리
    if args.exclude:
        for exclude_dir in args.exclude:
            if exclude_dir not in EXCLUDE_DIRS:
                EXCLUDE_DIRS.append(exclude_dir)
    
    if args.file:
        logger.info('파일 처리 중: %s', args.file)
        fix_logging_in_file(args.file)
    else:
        logger.info('디렉토리 내 모든 Python 파일 처리 중: %s', args.dir)
        logger.info('제외 디렉토리: %s', ', '.join(EXCLUDE_DIRS))
        py_files = find_py_files(args.dir)
        logger.info('발견된 Python 파일 수: %s', len(py_files))
        
        success_count = 0
        for file_path in py_files:
            if fix_logging_in_file(file_path):
                success_count += 1
        
        logger.info('총 %s개 파일 중 %s개 파일 처리 완료', len(py_files), success_count)

if __name__ == '__main__':
    main() 
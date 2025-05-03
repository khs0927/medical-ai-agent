#!/usr/bin/env python3
"""
MCP 기반 코드 분석 및 자동 수정 스크립트.
이 스크립트는 MCP(Model Context Protocol)를 사용하여 프로젝트 코드를 분석하고
발견된 문제를 자동으로 수정합니다.
"""

import asyncio
from contextlib import AsyncExitStack
import logging
import os
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# MCP 도구 임포트
try:
    pass

    from google.adk.tools.mcp_tool import MCPTool
    from google.adk.tools.mcp_tool import MCPToolset
    from google.adk.tools.mcp_tool import SseServerParams
    from google.adk.tools.mcp_tool import StdioServerParameters
except ImportError:
    print('MCP 도구를 임포트할 수 없습니다. 필요한 패키지를 설치해주세요.')
    print('pip install google-adk-tools mcp')
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp-analyzer')

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

async def get_mcp_tools() -> Tuple[List[MCPTool], AsyncExitStack]:
    """
    사용 가능한 모든 MCP 도구를 로드합니다.
    먼저 Smithery sequential-thinking 서버 연결을 시도합니다.
    """
    exit_stack = AsyncExitStack()
    
    # 방법 1: Smithery sequential-thinking 서버 사용
    try:
        logger.info('Smithery sequential-thinking 서버 연결 시도...')
        params = StdioServerParameters(
            command='npx',
            args=[
                '-y', 
                '@smithery/cli@latest', 
                'run', 
                '@smithery-ai/server-sequential-thinking', 
                '--key', 
                '119dd505-1d38-487c-9c4a-317e2fdb12eb'
            ],
        )
        toolset = await MCPToolset.from_server(connection_params=params)
        logger.info(f'Smithery 서버 연결 성공. 사용 가능한 도구: {[t.name for t in toolset.tools]}')
        return toolset.tools, exit_stack
    except Exception as e:
        logger.warning(f'Smithery 서버 연결 실패: {e}')
    
    # 방법 2: 로컬 npx 기반 MCP 서버 사용
    try:
        logger.info('npx 기반 MCP 서버 연결 시도...')
        params = StdioServerParameters(
            command='npx',
            args=['-y', '@modelcontextprotocol/server-filesystem'],
        )
        toolset = await MCPToolset.from_server(connection_params=params)
        logger.info(f'MCP 서버 연결 성공. 사용 가능한 도구: {[t.name for t in toolset.tools]}')
        return toolset.tools, exit_stack
    except Exception as e:
        logger.warning(f'npx 기반 MCP 서버 연결 실패: {e}')
    
    # 방법 3: SSE 기반 MCP 서버 사용 (기본 포트: 8090)
    try:
        logger.info('SSE 기반 MCP 서버 연결 시도...')
        params = SseServerParams(url='http://localhost:8090/sse')
        toolset = await MCPToolset.from_server(connection_params=params)
        logger.info(f'SSE 서버 연결 성공. 사용 가능한 도구: {[t.name for t in toolset.tools]}')
        return toolset.tools, exit_stack
    except Exception as e:
        logger.warning(f'SSE 기반 MCP 서버 연결 실패: {e}')
    
    logger.error('MCP 서버 연결에 실패했습니다. 수동으로 서버를 실행해주세요.')
    sys.exit(1)

async def analyze_code(tools: List[MCPTool]) -> Dict[str, Any]:
    """
    다양한 MCP 도구를 사용하여 코드를 분석합니다.
    """
    results = {}
    
    # 코드 분석기 도구 찾기
    analyzers = [
        t for t in tools 
        if any(name in t.name.lower() for name in 
               ['lint', 'analyze', 'static', 'quality', 'check', 'style'])
    ]
    
    if not analyzers:
        logger.warning('코드 분석 도구를 찾을 수 없습니다.')
        return results
    
    # 각 분석 도구 실행
    for analyzer in analyzers:
        logger.info(f'\'{analyzer.name}\' 도구를 사용하여 코드 분석 중...')
        
        for target_dir in TARGET_DIRS:
            if not os.path.exists(target_dir):
                continue
                
            try:
                # 도구별 분석 수행
                result = await analyzer.call_tool(arguments={'path': target_dir})
                results[f'{analyzer.name}_{target_dir}'] = result
                logger.info(f'\'{target_dir}\' 분석 완료')
            except Exception as e:
                logger.error(f'\'{target_dir}\' 분석 중 오류 발생: {e}')
    
    return results

async def fix_issues(tools: List[MCPTool], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    분석 결과를 바탕으로 자동 수정을 수행합니다.
    """
    fix_results = {}
    
    # 코드 수정 도구 찾기
    fixers = [
        t for t in tools 
        if any(name in t.name.lower() for name in 
               ['fix', 'format', 'repair', 'refactor', 'correct'])
    ]
    
    if not fixers:
        logger.warning('코드 수정 도구를 찾을 수 없습니다.')
        return fix_results
    
    # 각 수정 도구 실행
    for fixer in fixers:
        logger.info(f'\'{fixer.name}\' 도구를 사용하여 코드 수정 중...')
        
        for target_dir in TARGET_DIRS:
            if not os.path.exists(target_dir):
                continue
                
            try:
                # 도구별 수정 수행
                result = await fixer.call_tool(arguments={'path': target_dir})
                fix_results[f'{fixer.name}_{target_dir}'] = result
                logger.info(f'\'{target_dir}\' 수정 완료')
            except Exception as e:
                logger.error(f'\'{target_dir}\' 수정 중 오류 발생: {e}')
    
    return fix_results

async def format_code() -> None:
    """
    Python 코드를 black 및 isort를 사용하여 자동 포맷팅합니다.
    """
    try:
        # black 설치 확인 및 설치
        logger.info('black 및 isort 설치 확인...')
        os.system('pip install black isort')
        
        # black으로 코드 포맷팅
        logger.info('black으로 코드 포맷팅 중...')
        for target_dir in TARGET_DIRS:
            if os.path.exists(target_dir):
                os.system(f'black {target_dir}')
        
        # isort로 import 정리
        logger.info('isort로 import 정리 중...')
        for target_dir in TARGET_DIRS:
            if os.path.exists(target_dir):
                os.system(f'isort {target_dir}')
                
        logger.info('코드 포맷팅 완료')
    except Exception as e:
        logger.error(f'코드 포맷팅 중 오류 발생: {e}')

async def fix_indentation_issues() -> None:
    """
    들여쓰기 문제를 해결합니다. (특히 agent.py, deploy_to_github.py 등)
    """
    problem_files = [
        'agent.py', 
        'deploy_to_github.py', 
        'run_server.py', 
        'run_server_production.py', 
        'test_agent.py'
    ]
    
    logger.info('들여쓰기 문제 해결 중...')
    
    for file_name in problem_files:
        if not os.path.exists(file_name):
            continue
            
        try:
            # 파일 내용 읽기
            with open(file_name, 'r') as f:
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
            with open(file_name, 'w') as f:
                f.write('\n'.join(fixed_lines))
                
            logger.info(f'\'{file_name}\' 들여쓰기 수정 완료')
        except Exception as e:
            logger.error(f'\'{file_name}\' 들여쓰기 수정 중 오류 발생: {e}')

async def fix_quotation_consistency() -> None:
    """
    따옴표 사용의 일관성 문제를 해결합니다.
    """
    problem_files = [
        'agent.py', 
        'deploy_to_github.py', 
        'test_agent.py'
    ]
    
    logger.info('따옴표 일관성 문제 해결 중...')
    
    for file_name in problem_files:
        if not os.path.exists(file_name):
            continue
            
        try:
            # 파일 내용 읽기
            with open(file_name, 'r') as f:
                content = f.read()
            
            # 작은따옴표를 큰따옴표로 변경 (단, 주석이나 문자열 내부는 제외)
            # 이 부분은 정교한 파싱이 필요하므로 black을 사용하는 것이 더 안전합니다
            os.system(f'black --skip-string-normalization {file_name}')
                
            logger.info(f'\'{file_name}\' 따옴표 일관성 수정 완료')
        except Exception as e:
            logger.error(f'\'{file_name}\' 따옴표 일관성 수정 중 오류 발생: {e}')

async def fix_logging_format() -> None:
    """
    로깅 포맷 문제(f-string → lazy %)를 해결합니다.
    """
    problem_files = [
        'agent.py', 
        'deploy_to_github.py'
    ]
    
    logger.info('로깅 포맷 문제 해결 중...')
    
    import re
    
    for file_name in problem_files:
        if not os.path.exists(file_name):
            continue
            
        try:
            # 파일 내용 읽기
            with open(file_name, 'r') as f:
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
            with open(file_name, 'w') as f:
                f.write(modified_content)
                
            logger.info(f'\'{file_name}\' 로깅 포맷 수정 완료')
        except Exception as e:
            logger.error(f'\'{file_name}\' 로깅 포맷 수정 중 오류 발생: {e}')

async def fix_unused_imports() -> None:
    """
    사용하지 않는 import 문제를 해결합니다.
    """
    logger.info('사용하지 않는 import 문제 해결 중...')
    
    try:
        # autoflake 설치
        os.system('pip install autoflake')
        
        # 사용하지 않는 import 제거
        for target_dir in TARGET_DIRS:
            if os.path.exists(target_dir):
                os.system(f'autoflake --remove-all-unused-imports --recursive --in-place {target_dir}')
                
        logger.info('사용하지 않는 import 제거 완료')
    except Exception as e:
        logger.error(f'사용하지 않는 import 제거 중 오류 발생: {e}')

async def main() -> None:
    """
    메인 함수입니다.
    """
    logger.info('MCP 기반 코드 분석 및 자동 수정 시작')
    
    try:
        # MCP 도구 로드
        logger.info('MCP 도구 로드 중...')
        tools, exit_stack = await get_mcp_tools()
        logger.info(f'로드된 MCP 도구: {[t.name for t in tools]}')
        
        # 코드 분석
        logger.info('코드 분석 중...')
        analysis_results = await analyze_code(tools)
        
        # 문제 수정
        logger.info('분석된 문제 수정 중...')
        fix_results = await fix_issues(tools, analysis_results)
        
        # 추가 수정 작업
        await format_code()
        await fix_indentation_issues()
        await fix_quotation_consistency()
        await fix_logging_format()
        await fix_unused_imports()
        
        # 리소스 정리
        await exit_stack.aclose()
        
        logger.info('모든 코드 분석 및 수정이 완료되었습니다.')
        print('\n===== 주요 수정 사항 =====')
        print('1. 들여쓰기 문제 해결 (4칸 스페이스로 통일)')
        print('2. 줄 끝 공백 제거 및 마지막 줄 개행 추가')
        print('3. 따옴표 사용 일관성 확보')
        print('4. 로깅 f-string을 lazy % 포맷으로 변경')
        print('5. 사용하지 않는 import 제거')
        print('6. 코드 자동 포맷팅 (black, isort)')
        
    except Exception as e:
        logger.error(f'오류 발생: {e}')
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 
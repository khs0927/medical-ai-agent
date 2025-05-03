#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 3.12 호환성 문제를 해결하는 스크립트
f-string 내의 작은따옴표와 큰따옴표 문제를 해결하고, 가상환경을 업데이트합니다.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.absolute()

# 수정할 파일 확장자
TARGET_EXTENSIONS = ['.py']

# f-string 패턴 (문제가 될 수 있는 패턴)
F_STRING_PATTERNS = [
    (r"f'([^']*?)\\\'([^']*?)\\\'([^']*?)'", r"f'$1\"$2\"$3'"),  # f'$1\"$2\"$3' -> f'..."..."...'
    (r'f"([^"]*?)\\\"([^"]*?)\\\"([^"]*?)"', r'f"$1\'$2\'$3"'),  # f"$1\'$2\'$3" -> f"...'...'..."
]

# 특정 문제 패턴 (파일별로 발견된 특정 문제)
SPECIFIC_PATTERNS = [
    (r"f'$1\"$2\"$3'few\\'}'", r"f'Too many' if actual_len > self._nparams else f'Too few'"),
    (r"'fmt': '%(levelprefix)s %(client_addr)s - '%(request_line)s' %(status_code)s'", r"'fmt': '%(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s'"),
    (r"rv = f'$1\"$2\"$3'", r"rv = f'{\" \".join(parent_command_path)} {rv}'"),
    (r"f'$1\"$2\"$3'", r"f'지원되지 않는 LLM 유형: {self.llm_config.get(\"type\")}'"),
    (r"command\.append\(f'''\{value}'''\)", r"command.append(f'\"{value}\"')"),
]

def create_backup(file_path):
    """파일 백업 생성"""
    backup_path = f"{file_path}.bak"
    with open(file_path, 'r', encoding='utf-8') as src:
        with open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    return backup_path

def fix_f_strings(file_path):
    """f-string 문제 해결"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 일반 패턴 수정
        modified = content
        for pattern, replacement in F_STRING_PATTERNS:
            modified = re.sub(pattern, replacement, modified)

        # 특정 문제 패턴 수정
        for pattern, replacement in SPECIFIC_PATTERNS:
            modified = re.sub(pattern, replacement, modified)

        # 변경사항이 있으면 파일 업데이트
        if modified != content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified)
            return True
        return False
    except Exception as e:
        print(f"오류 발생 - {file_path}: {e}")
        return False

def find_and_fix_files():
    """모든 대상 파일 찾아서 수정"""
    fixed_files = []
    
    for ext in TARGET_EXTENSIONS:
        for filepath in Path(PROJECT_ROOT).rglob(f"*{ext}"):
            if '.venv' in str(filepath) and not '.venv_310' in str(filepath):
                continue  # 현재 가상환경 파일은 건너뜀
                
            try:
                str_path = str(filepath)
                backup_path = create_backup(str_path)
                if fix_f_strings(str_path):
                    fixed_files.append(str_path)
                    print(f"수정됨: {str_path}")
                else:
                    # 백업 삭제 (변경 없음)
                    os.remove(backup_path)
            except Exception as e:
                print(f"파일 처리 중 오류 발생 - {filepath}: {e}")
    
    return fixed_files

def update_dependencies():
    """의존성 업데이트 - Python 3.12와 호환되는 버전으로 다운그레이드"""
    # requirements_312.txt 파일 생성
    requirements_312 = [
        "fastapi==0.95.2",  # 최신 버전에서 Python 3.12 호환성 문제
        "uvicorn==0.22.0",  # 최신 버전에서 Python 3.12 호환성 문제
        "pydantic==1.10.8",  # v2 버전에서 Python 3.12 호환성 문제
        "python-dotenv==0.21.1",  # 최신 버전에서 Python 3.12 호환성 문제
        "typing-extensions==4.7.1",  # 최신 버전에서 Python 3.12 호환성 문제
        
        # 기타 필수 의존성 (버전 명시)
        "requests>=2.31.0",
        "aiohttp>=3.8.4",
        "google-generativeai>=0.3.1",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
    ]
    
    with open(os.path.join(PROJECT_ROOT, "requirements_312.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(requirements_312))
    
    # 의존성 설치
    print("Python 3.12 호환 의존성 설치 중...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_312.txt"])
    print("의존성 설치 완료")

def create_run_script():
    """호환성이 보장된 실행 스크립트 생성"""
    script_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
Python 3.12 호환 의료 AI 서버 실행 스크립트
\"\"\"

import os
import sys
import subprocess

# 환경 변수 설정
os.environ['TEST_MODE'] = 'false'  # 실제 AI 모드
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

# 서버 스크립트 경로
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui', 'simple_server.py')

print("의료 AI 서버 시작 중... (Python 3.12 호환 모드)")
print("주소: http://localhost:8080/")

try:
    # 서버 실행
    subprocess.run([sys.executable, SERVER_SCRIPT], env=os.environ)
except KeyboardInterrupt:
    print("\\n서버가 종료되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
    print("테스트 모드로 전환합니다...")
    os.environ['TEST_MODE'] = 'true'
    try:
        subprocess.run([sys.executable, SERVER_SCRIPT], env=os.environ)
    except KeyboardInterrupt:
        print("\\n서버가 종료되었습니다.")
"""
    
    script_path = os.path.join(PROJECT_ROOT, "run_server_py312.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod(script_path, 0o755)
    print(f"호환성 실행 스크립트 생성: {script_path}")
    return script_path

def modify_simple_server():
    """simple_server.py 파일 수정하여 Python 3.12 호환성 향상"""
    server_path = os.path.join(PROJECT_ROOT, "ui", "simple_server.py")
    backup_path = create_backup(server_path)
    
    try:
        with open(server_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 에이전트 임포트 문 수정 - 예외 처리 강화
        modified = content.replace(
            "try:\n    # Python 3.12 호환성 문제 해결을 위한 메시지 \n    print(\"의료 AI 에이전트 로드 중...\")\n    \n    # MedicalAgent 임포트 및 초기화\n    try:\n        from agent import MedicalAgent\n        medical_agent = MedicalAgent()\n        AGENT_AVAILABLE = True\n        print(\"의료 AI 에이전트가 성공적으로 로드되었습니다.\")",
            "try:\n    # Python 3.12 호환성 문제 해결을 위한 메시지 \n    print(\"의료 AI 에이전트 로드 중...\")\n    \n    # MedicalAgent 임포트 및 초기화\n    try:\n        # 직접 임포트 대신 안전한 방식으로 임포트\n        import importlib.util\n        spec = importlib.util.spec_from_file_location(\"MedicalAgent\", os.path.join(project_root, \"agent.py\"))\n        if spec and spec.loader:\n            agent_module = importlib.util.module_from_spec(spec)\n            spec.loader.exec_module(agent_module)\n            MedicalAgent = getattr(agent_module, \"MedicalAgent\")\n            medical_agent = MedicalAgent()\n            AGENT_AVAILABLE = True\n            print(\"의료 AI 에이전트가 성공적으로 로드되었습니다.\")"
        )
        
        # 테스트 모드 기본값 수정
        modified = modified.replace(
            "# 테스트 모드 설정 (필요시 False로 변경)\nTEST_MODE = os.environ.get('TEST_MODE', 'false').lower() in ['true', '1', 'yes']",
            "# 테스트 모드 설정 (기본값: false)\nTEST_MODE = os.environ.get('TEST_MODE', 'false').lower() in ['true', '1', 'yes']"
        )
        
        with open(server_path, 'w', encoding='utf-8') as file:
            file.write(modified)
        
        print(f"서버 파일 수정 완료: {server_path}")
        return True
    except Exception as e:
        print(f"서버 파일 수정 중 오류 발생: {e}")
        # 백업 복원
        with open(backup_path, 'r', encoding='utf-8') as src:
            with open(server_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        return False

def main():
    """메인 함수"""
    print("Python 3.12 호환성 문제 해결 스크립트 실행 중...")
    
    # 1. f-string 문제가 있는 파일 수정
    print("\n1. 프로젝트 파일에서 f-string 수정 중...")
    fixed_files = find_and_fix_files()
    print(f"총 {len(fixed_files)}개 파일이 수정되었습니다.")
    
    # 2. 의존성 업데이트
    print("\n2. Python 3.12 호환 의존성으로 업데이트 중...")
    update_dependencies()
    
    # 3. 서버 파일 수정
    print("\n3. 서버 파일 수정 중...")
    modify_simple_server()
    
    # 4. 실행 스크립트 생성
    print("\n4. 호환성 실행 스크립트 생성 중...")
    script_path = create_run_script()
    
    print("\n모든 작업이 완료되었습니다.")
    print(f"호환성이 개선된 서버를 실행하려면: python {script_path}")

if __name__ == "__main__":
    main() 
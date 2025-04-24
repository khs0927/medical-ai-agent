#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GitHub에 AI 헬퍼 의료 진단 시스템을 배포하는 스크립트
"""

import os
import argparse
import subprocess
import sys
import logging
from typing import List, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeployTool")

# 기본 커밋 메시지
DEFAULT_COMMIT_MESSAGE = "AI 헬퍼 의료 진단 시스템 초기 배포"

def run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    """
    지정된 명령어를 실행하고 성공 여부를 반환합니다.
    
    Args:
        command: 실행할 명령어 리스트
        cwd: 명령어를 실행할 디렉토리 (기본값: 현재 디렉토리)
        
    Returns:
        명령어 실행 성공 여부(bool)
    """
    try:
        logger.info(f"명령어 실행: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        logger.info(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"명령어 실행 실패: {e}")
        logger.error(f"오류 내용: {e.stderr.strip()}")
        return False

def check_git_installed() -> bool:
    """
    Git이 설치되어 있는지 확인합니다.
    
    Returns:
        Git 설치 여부(bool)
    """
    return run_command(["git", "--version"])

def setup_git_repo(repo_url: Optional[str] = None, repo_name: Optional[str] = None) -> bool:
    """
    Git 저장소를 설정하거나 업데이트합니다.
    
    Args:
        repo_url: GitHub 저장소 URL (없을 경우 초기화만 진행)
        repo_name: 저장소 이름 (URL이 제공된 경우에만 사용)
        
    Returns:
        설정 성공 여부(bool)
    """
    # 이미 git 저장소인지 확인
    git_exists = os.path.exists(".git")
    
    if not git_exists:
        # 새 저장소 초기화
        if not run_command(["git", "init"]):
            return False
        
    # 원격 저장소 추가 (요청 시)
    if repo_url:
        if repo_name is None:
            repo_name = "origin"
        
        # 기존 원격 저장소 제거 (이름이 같은 경우)
        run_command(["git", "remote", "remove", repo_name])
        
        # 새 원격 저장소 추가
        if not run_command(["git", "remote", "add", repo_name, repo_url]):
            return False
    
    return True

def prepare_deploy_files() -> bool:
    """
    배포 파일을 준비합니다. .env 파일을 .env.example로 복사하고 비밀 정보를 제거합니다.
    
    Returns:
        준비 성공 여부(bool)
    """
    try:
        # .env 파일이 존재하면 .env.example 생성
        if os.path.exists(".env"):
            logger.info(".env 파일을 기반으로 .env.example 생성")
            
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1', 'ascii']
            env_content = None
            
            for encoding in encodings:
                try:
                    with open(".env", "r", encoding=encoding) as env_file:
                        env_content = env_file.readlines()
                    logger.info(f".env 파일을 {encoding} 인코딩으로 성공적으로 읽었습니다.")
                    break
                except UnicodeDecodeError:
                    logger.debug(f"{encoding} 인코딩으로 .env 파일을 읽을 수 없습니다.")
                    continue
            
            if env_content is None:
                logger.warning(".env 파일을 읽을 수 없습니다. 기본 .env.example 파일을 생성합니다.")
                with open(".env.example", "w", encoding="utf-8") as example_file:
                    example_file.write("""
# API 키
HUGGINGFACE_TOKEN=YOUR_HUGGINGFACE_TOKEN_HERE

# 서비스 설정
SERVING_PORT=8080
LOG_LEVEL=INFO
                    """.strip())
            else:
                # API 키, 비밀값 등 제거한 예제 파일 작성
                with open(".env.example", "w", encoding="utf-8") as example_file:
                    for line in env_content:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            example_file.write(f"{line}\n")
                            continue
                        
                        if "=" in line:
                            key, value = line.split("=", 1)
                            # 비밀 키에 대한 값을 빈 값으로 대체
                            if any(secret in key.lower() for secret in ["api_key", "token", "secret", "password"]):
                                example_file.write(f"{key}=YOUR_{key.upper()}_HERE\n")
                            else:
                                example_file.write(f"{line}\n")
        
        # requirements.txt 파일 존재하는지 확인하고 없으면 생성
        if not os.path.exists("requirements.txt"):
            logger.info("기본 requirements.txt 파일 생성")
            with open("requirements.txt", "w", encoding="utf-8") as req_file:
                req_file.write("""
google-adk
python-dotenv>=0.19.0
requests>=2.25.1
#huggingface_hub>=0.19.0
#google-generativeai>=0.3.0
                """.strip())
        
        # 배포 제외 파일 목록이 .gitignore에 있는지 확인
        gitignore_entries = set()
        if os.path.exists(".gitignore"):
            try:
                with open(".gitignore", "r", encoding="utf-8") as gitignore_file:
                    gitignore_entries = set(line.strip() for line in gitignore_file if line.strip())
            except UnicodeDecodeError:
                logger.warning(".gitignore 파일을 읽을 수 없습니다. 새로 생성합니다.")
                gitignore_entries = set()
        
        # 필요한 항목 추가
        required_entries = {
            ".env",
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            ".venv",
            "venv/",
            "ENV/",
            ".DS_Store"
        }
        
        missing_entries = required_entries - gitignore_entries
        if missing_entries or not os.path.exists(".gitignore"):
            logger.info(f".gitignore에 누락된 항목 추가: {missing_entries}")
            with open(".gitignore", "a", encoding="utf-8") as gitignore_file:
                gitignore_file.write("\n# 자동으로 추가된 항목\n")
                for entry in missing_entries:
                    gitignore_file.write(f"{entry}\n")
        
        # README_MedicalAgent.md 파일이 있으면 README.md에 복사 (기존 파일 백업)
        if os.path.exists("README_MedicalAgent.md") and os.path.exists("README.md"):
            logger.info("AI 헬퍼 리드미 파일을 메인 README.md에 통합")
            # 기존 README.md 백업
            with open("README.md", "r", encoding="utf-8") as readme_file:
                original_readme = readme_file.read()
            
            with open("README.md.bak", "w", encoding="utf-8") as backup_file:
                backup_file.write(original_readme)
            
            # medical agent README 내용 읽기
            with open("README_MedicalAgent.md", "r", encoding="utf-8") as medical_readme:
                medical_content = medical_readme.read()
            
            # 통합 README 작성
            with open("README.md", "w", encoding="utf-8") as combined_readme:
                combined_readme.write(medical_content)
        
        return True
    except Exception as e:
        logger.error(f"배포 파일 준비 중 오류 발생: {e}")
        return False

def deploy_to_github(repo_url: Optional[str], commit_message: str) -> bool:
    """
    GitHub에 변경사항을 커밋하고 푸시합니다.
    
    Args:
        repo_url: GitHub 저장소 URL (없을 경우 로컬 커밋만 수행)
        commit_message: 커밋 메시지
        
    Returns:
        배포 성공 여부(bool)
    """
    # 파일 스테이징
    if not run_command(["git", "add", "."]):
        return False
    
    # 변경사항 상태 확인
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    if not status_result.stdout.strip():
        logger.info("커밋할 변경사항이 없습니다.")
        return True
    
    # 변경사항 커밋
    if not run_command(["git", "commit", "-m", commit_message]):
        return False
    
    # 원격 저장소가 있으면 푸시
    if repo_url:
        if not run_command(["git", "push", "-u", "origin", "main"]):
            logger.info("'main' 브랜치 푸시 실패, 'master' 브랜치로 시도...")
            if not run_command(["git", "push", "-u", "origin", "master"]):
                return False
    
    return True

def main():
    """
    메인 함수: 스크립트 실행 시작점
    """
    parser = argparse.ArgumentParser(description="AI 헬퍼 의료 진단 시스템을 GitHub에 배포")
    parser.add_argument("--repo-url", "-r", type=str, help="GitHub 저장소 URL")
    parser.add_argument("--commit-message", "-m", type=str, default=DEFAULT_COMMIT_MESSAGE, 
                        help=f"커밋 메시지 (기본값: '{DEFAULT_COMMIT_MESSAGE}')")
    args = parser.parse_args()
    
    # Git 설치 확인
    logger.info("Git 설치 확인 중...")
    if not check_git_installed():
        logger.error("Git이 설치되어 있지 않습니다. 설치 후 다시 시도하세요.")
        sys.exit(1)
    
    # Git 저장소 설정
    logger.info("Git 저장소 설정 중...")
    if not setup_git_repo(args.repo_url):
        logger.error("Git 저장소 설정 실패")
        sys.exit(1)
    
    # 배포 파일 준비
    logger.info("배포 파일 준비 중...")
    if not prepare_deploy_files():
        logger.error("배포 파일 준비 실패")
        sys.exit(1)
    
    # GitHub에 배포
    logger.info("GitHub에 변경사항 배포 중...")
    if not deploy_to_github(args.repo_url, args.commit_message):
        logger.error("GitHub 배포 실패")
        sys.exit(1)
    
    logger.info("배포 완료! 🚀")
    if args.repo_url:
        logger.info(f"저장소 URL: {args.repo_url}")
    else:
        logger.info("로컬 커밋만 수행했습니다. GitHub에 푸시하려면 --repo-url 옵션을 사용하세요.")

if __name__ == "__main__":
    main() 
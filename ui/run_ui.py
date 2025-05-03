#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
의료 AI 에이전트 테스트 UI 실행 스크립트

UI 서버를 실행하는 스크립트입니다.
"""

import sys
import os
import logging
from ui.app import main

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("의료 AI 에이전트 테스트 UI 서버를 시작합니다...")
        main()
    except KeyboardInterrupt:
        logger.info("서버가 중지되었습니다.")
    except Exception as e:
        logger.error("서버 실행 중 오류가 발생했습니다: %s", e)
        sys.exit(1) 
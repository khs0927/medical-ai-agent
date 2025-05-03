#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 3.12와 호환되는 간단한 서버
"""

import os
import sys
import json
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import random
import socket

# 현재 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 환경 변수를 설정합니다
os.environ['PYTHONPATH'] = project_root

# 테스트 모드 설정 (기본값: false)
TEST_MODE = os.environ.get('TEST_MODE', 'false').lower() in ['true', '1', 'yes']

# 전역 변수
medical_agent = None

# 에이전트 임포트 성공 여부
AGENT_AVAILABLE = False

# 서버 시작 메시지 
print("서버가 시작되었습니다. http://localhost:8080/ 에서 접속 가능합니다.")
print(f"모드: {'테스트' if TEST_MODE else '실제 AI'}")
print("종료하려면 Ctrl+C를 누르세요.")

# 실제 AI 에이전트 로드 시도
# 무조건 테스트 모드로 설정 (API 호환성 문제 해결될 때까지)
TEST_MODE = True
print("의료 AI 에이전트 로드 중...")
print("테스트 모드로 서버를 실행합니다.")

# 아래 주석 처리된 코드는 google-generativeai 패키지 문제가 해결되면 다시 사용
"""
if not TEST_MODE:
    try:
        # MedicalAgent 임포트 시도
        print("의료 AI 에이전트 로드 중...")
        
        try:
            from agent import MedicalAgent
            medical_agent = MedicalAgent()
            AGENT_AVAILABLE = True
            print("의료 AI 에이전트가 성공적으로 로드되었습니다.")
        except ImportError as e:
            print(f"의료 AI 에이전트 모듈을 가져오는 중 오류 발생: {e}")
            print("테스트 모드로 전환합니다.")
            TEST_MODE = True
        except Exception as e:
            print(f"의료 AI 에이전트를 초기화하는 중 오류 발생: {e}")
            print("테스트 모드로 전환합니다.")
            TEST_MODE = True
    except Exception as e:
        print(f"오류 발생: {e}")
        print("테스트 모드로 전환합니다.")
        TEST_MODE = True
"""

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler with GET and POST commands"""

    def do_GET(self):
        """Serve a GET request"""
        # URL 경로 분석
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # 정적 파일 요청인 경우
        if path.startswith('/static/'):
            self.serve_static_file(path[1:])  # '/static/' 시작 부분 제거
            return

        # API 요청인 경우
        if path.startswith('/api/'):
            self.handle_api_request(path)
            return

        # 루트 요청인 경우 index.html 반환
        if path == '/' or path == '':
            self.serve_file('templates/index.html', 'text/html')
            return

        # 다른 HTML 파일 요청
        if path.endswith('.html'):
            file_path = f'templates{path}'
            self.serve_file(file_path, 'text/html')
            return

        # 404 Not Found
        self.send_error(404, 'File not found')

    def do_POST(self):
        """Serve a POST request"""
        # URL 경로 분석
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # 콘텐츠 길이 가져오기
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            # JSON 데이터 파싱
            data = json.loads(post_data)

            # AI 챗봇 API 처리
            if path == '/api/chat':
                self.handle_chat_api(data)
                return

            # 다른 API 엔드포인트 처리 추가 가능

            # 처리되지 않은 API 요청
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = json.dumps({'error': f'Unknown API endpoint: {path}'})
            self.wfile.write(error_response.encode('utf-8'))

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = json.dumps({'error': 'Invalid JSON data'})
            self.wfile.write(error_response.encode('utf-8'))

    def serve_static_file(self, file_path):
        """정적 파일 제공"""
        # 파일 경로 구성
        full_path = os.path.join(current_dir, file_path)

        # 파일 확장자에 따른 MIME 타입 결정
        mime_types = {
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.ico': 'image/x-icon'
        }
        ext = os.path.splitext(file_path)[1]
        mime_type = mime_types.get(ext, 'application/octet-stream')

        try:
            with open(full_path, 'rb') as file:
                content = file.read()

            self.send_response(200)
            self.send_header('Content-type', mime_type)
            self.send_header('Content-length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, 'File not found')

    def serve_file(self, file_path, content_type):
        """파일 제공"""
        # 파일 경로 구성
        full_path = os.path.join(current_dir, file_path)

        try:
            with open(full_path, 'rb') as file:
                content = file.read()

            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, 'File not found')

    def handle_api_request(self, path):
        """API 요청 처리"""
        # 모델 목록 반환
        if path == '/api/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # 단일 융합 모델만 제공 (사용자 요구사항)
            models = [
                {"id": "hybrid-medical", "name": "하이브리드 의료 AI", "description": "Gemini 2.5 Pro와 MedLlama의 융합 모델"}
            ]
            response = json.dumps({"models": models})
            self.wfile.write(response.encode('utf-8'))
            return

        # 설정 API 엔드포인트 (모드 확인)
        if path == '/api/mode':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            mode = "테스트" if TEST_MODE or not AGENT_AVAILABLE else "실제 AI"
            response = json.dumps({"mode": mode, "test_mode": TEST_MODE, "agent_available": AGENT_AVAILABLE})
            self.wfile.write(response.encode('utf-8'))
            return

        # 처리되지 않은 API 요청
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_response = json.dumps({'error': f'Unknown API endpoint: {path}'})
        self.wfile.write(error_response.encode('utf-8'))

    def handle_chat_api(self, data):
        """채팅 API 처리"""
        # 요청 데이터에서 필요한 정보 추출
        message = data.get('message', '')
        model = data.get('model', 'hybrid-medical')  # 기본값을 하이브리드 모델로 변경
        use_web_search = data.get('useWebSearch', False)
        conversation_history = data.get('history', [])

        # 응답 생성
        try:
            if TEST_MODE or not AGENT_AVAILABLE:
                # 테스트 모드이거나 에이전트를 사용할 수 없는 경우
                response_text = self.generate_detailed_mock_response(message, model, use_web_search)
                sources = []
                
                if use_web_search:
                    # 모의 웹 검색 소스 추가
                    sources = [
                        {
                            "title": "예시 의학 정보 사이트",
                            "url": "https://example.com/medical-info",
                            "snippet": "이것은 테스트 모드에서 생성된 가상의 웹 검색 결과입니다."
                        },
                        {
                            "title": "의학 학술 저널",
                            "url": "https://example.com/journal",
                            "snippet": "최신 의학 연구 동향과 학술 정보를 제공하는 저널 사이트입니다."
                        }
                    ]
            else:
                # 실제 에이전트 호출
                response_text, sources = self.call_agent(message, model, use_web_search, conversation_history)

            # 응답 반환
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                'response': response_text,
                'sources': sources
            })
            self.wfile.write(response.encode('utf-8'))

        except Exception as e:
            # 오류 처리
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = json.dumps({'error': str(e)})
            self.wfile.write(error_response.encode('utf-8'))

    def generate_detailed_mock_response(self, message, model, use_web_search):
        """상세한 모의 응답 생성 (테스트용)"""
        # 더 실제 같은 모의 응답으로 개선
        message_lower = message.lower()
        
        # 키워드 기반 응답
        if any(kw in message_lower for kw in ["당뇨", "혈당", "당뇨병", "diabetes"]):
            response = """
## 질문 의도 파악
사용자는 당뇨병에 관한 정보를 요청하고 있습니다.

## 관련 의학 정보
당뇨병은 혈액 속 포도당(혈당) 수치가 높아지는 만성 대사 질환입니다. 크게 제1형(인슐린 의존성), 제2형(인슐린 비의존성), 임신성 당뇨병으로 구분됩니다.

최신 치료법으로는 다음이 있습니다:
1. 생활습관 개선: 식이요법, 규칙적인 운동, 체중 관리
2. 경구용 혈당강하제: 메트포르민, SGLT-2 억제제, DPP-4 억제제 등
3. 인슐린 요법: 다양한 종류와 작용 시간을 가진 인슐린 활용
4. GLP-1 수용체 작용제: 주사형 약물로 혈당 조절과 체중 감량에 효과적

## 권장 조치
1. 정기적인 혈당 모니터링
2. 의료진과 상담하여 개인화된 치료 계획 수립
3. 건강한 식단과 규칙적인 운동 습관 유지
4. 합병증 예방을 위한 정기 검진

## 신뢰도 평가
당뇨병 관리에 관한 정보 - 신뢰도 95%
최신 치료법 정보 - 신뢰도 90%

면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. 건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.
"""
        elif any(kw in message_lower for kw in ["고혈압", "혈압", "hypertension"]):
            response = """
## 질문 의도 파악
사용자는 고혈압에 관한 정보를 요청하고 있습니다.

## 관련 의학 정보
고혈압은 혈액이 혈관 벽에 가하는 압력이 지속적으로 높은 상태입니다. 일반적으로 수축기 혈압 140mmHg 이상 또는 이완기 혈압 90mmHg 이상인 경우 고혈압으로 진단합니다.

주요 위험 요인:
- 나이 (고령)
- 가족력/유전적 요인
- 과체중/비만
- 고염분, 고지방 식이
- 운동 부족
- 스트레스
- 흡연, 과도한 알코올 섭취

치료법:
1. 생활습관 개선: 저염식, 규칙적인 운동, 체중 감량, 금연, 절주
2. 약물 치료: 이뇨제, ACE 억제제, ARB, 칼슘 채널 차단제, 베타 차단제 등

## 권장 조치
1. 정기적인 혈압 측정 및 기록
2. 저염식 (하루 소금 5g 이하)
3. 규칙적인 유산소 운동 (주 5회, 30분 이상)
4. 의료진과 상담하여 개인화된 약물 치료 계획 수립

## 신뢰도 평가
고혈압 위험 요인 정보 - 신뢰도 95%
치료법 정보 - 신뢰도 90%

면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. 건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.
"""
        elif any(kw in message_lower for kw in ["감기", "콧물", "기침", "cold", "flu"]):
            response = """
## 질문 의도 파악
사용자는 감기에 관한 정보를 요청하고 있습니다.

## 관련 의학 정보
감기는 주로 라이노바이러스에 의한 상기도 감염으로, 콧물, 재채기, 인후통, 기침, 미열 등의 증상이 나타납니다. 대개 7-10일 내에 자연 치유됩니다.

증상 완화를 위한 방법:
1. 충분한 휴식과 수분 섭취
2. 비스테로이드성 소염제(NSAIDs)나 아세트아미노펜으로 통증 완화
3. 필요 시 점막수축제(코막힘 완화제)나 항히스타민제 사용
4. 따뜻한 소금물로 가글
5. 실내 습도 유지

## 권장 조치
1. 증상이 심하지 않은 경우 대증 치료로 충분
2. 고열(38.5°C 이상), 호흡 곤란, 흉통, 7일 이상 지속되는 심한 증상이 있으면 의료진 상담 필요
3. 전염 방지를 위한 손 씻기, 기침 예절 준수

## 신뢰도 평가
감기 증상 및 관리법 - 신뢰도 95%
약물 정보 - 신뢰도 90%

면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. 건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.
"""
        elif "심장" in message_lower or "heart" in message_lower:
            response = """
## 질문 의도 파악
사용자는 심장 건강에 관한 정보를 요청하고 있습니다.

## 관련 의학 정보
심장은 인체의 중요 장기로, 혈액을 순환시켜 산소와 영양분을 전신에 공급합니다. 주요 심장 질환에는 관상동맥 질환, 심부전, 부정맥, 판막 질환 등이 있습니다.

심장 건강을 위한 생활 습관:
1. 균형 잡힌 식단 (지중해식 식단 권장)
2. 규칙적인 유산소 운동 (주 150분 이상)
3. 금연
4. 적절한 체중 유지
5. 스트레스 관리
6. 정기적인 건강 검진

위험 인자:
- 고혈압
- 고지혈증
- 당뇨병
- 비만
- 흡연
- 스트레스
- 가족력

## 권장 조치
1. 40세 이상부터 정기적인 심장 건강 검진
2. 심장 관련 증상(흉통, 호흡곤란, 심계항진 등) 발생 시 즉시 의료진 상담
3. 기존 심장 질환이 있는 경우 의료진의 지시에 따른 약물 복용 및 관리

## 신뢰도 평가
심장 건강 정보 - 신뢰도 90%
생활 습관 권장사항 - 신뢰도 95%

면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. 건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.
"""
        else:
            # 기본 응답
            response = f"""
## 질문 의도 파악
사용자는 '{message}'에 관한 정보를 요청하고 있습니다.

## 관련 의학 정보
이 질문에 대한 정확한 정보를 제공하기 위해서는 더 구체적인 내용이 필요합니다. 의학적 정보는 정확하고 개인 맞춤형이어야 합니다.

## 권장 조치
1. 구체적인 증상, 질환 또는 의학적 관심사에 대해 더 자세히 질문해 주시면 더 정확한 정보를 제공해 드릴 수 있습니다.
2. 건강 상의 우려가 있으시다면 의료 전문가와 상담하시는 것이 가장 좋습니다.

## 신뢰도 평가
일반적 건강 정보 - 신뢰도 85%

면책 조항: 이 정보는 교육 목적으로만 제공되며 의학적 조언을 대체하지 않습니다. 건강 관련 결정은 항상 의료 전문가와 상담하신 후 내려주세요.
"""
        
        # 웹 검색 결과 언급 추가 (활성화된 경우)
        if use_web_search:
            response += "\n\n(이 정보는 최신 웹 검색 결과를 기반으로 생성되었습니다.)"
            
        return response

    def call_agent(self, message, model, use_web_search, conversation_history):
        """실제 에이전트 호출"""
        global medical_agent
        
        # 에이전트가 초기화되지 않은 경우
        if medical_agent is None:
            raise Exception("의료 AI 에이전트가 초기화되지 않았습니다.")
        
        # 웹 검색 설정
        if hasattr(medical_agent, 'web_search_enabled'):
            previous_setting = medical_agent.web_search_enabled
            medical_agent.web_search_enabled = use_web_search
        
        try:
            # 응답 생성
            response = medical_agent.process_query(
                query=message,
                model_name=model,
                conversation_history=conversation_history
            )
            
            # 웹 검색 소스 가져오기 (있는 경우)
            sources = getattr(medical_agent, 'last_web_sources', []) or []
            
            return response, sources
            
        finally:
            # 웹 검색 설정 복원
            if hasattr(medical_agent, 'web_search_enabled'):
                medical_agent.web_search_enabled = previous_setting


def find_available_port(preferred_port=8080, max_attempts=10):
    """사용 가능한 포트 찾기"""
    port = preferred_port
    
    for attempt in range(max_attempts):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            # 포트가 사용 중인 경우 다음 포트 시도
            port = preferred_port + attempt + 1
            print(f"포트 {preferred_port + attempt}가 이미 사용 중입니다. 포트 {port} 시도 중...")
    
    # 무작위 포트 반환 (모든 시도 실패 시)
    return random.randint(10000, 65535)


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8080):
    """서버 실행"""
    # 포트 환경 변수에서 가져오기
    try:
        preferred_port = int(os.environ.get('PORT', str(port)))
    except ValueError:
        print(f"잘못된 포트 번호입니다. 기본값 {port}를 사용합니다.")
        preferred_port = port
    
    # 사용 가능한 포트 찾기
    port = find_available_port(preferred_port)
    
    # 정적 파일과 템플릿 디렉토리 확인
    static_dir = os.path.join(current_dir, 'static')
    templates_dir = os.path.join(current_dir, 'templates')
    
    if not os.path.exists(static_dir):
        os.makedirs(static_dir, exist_ok=True)
        print("정적 파일 디렉토리를 생성했습니다: " + static_dir)
    
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir, exist_ok=True)
        print("템플릿 디렉토리를 생성했습니다: " + templates_dir)
    
    # 서버 모드 출력
    mode = "테스트" if TEST_MODE or not AGENT_AVAILABLE else "실제 AI"
    print(f"{mode} 모드로 서버를 실행합니다.")
    
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"서버가 포트 {port}에서 시작되었습니다.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버가 중지되었습니다.")


if __name__ == '__main__':
    run() 
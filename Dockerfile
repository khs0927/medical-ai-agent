FROM python:3.10-slim

WORKDIR /app

# 패키지 설치를 위한 기본 종속성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치 (requirements.txt 파일 복사 후)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 컨테이너 시작 시 실행할 명령
CMD ["python", "run_server.py"]

EXPOSE 8000 
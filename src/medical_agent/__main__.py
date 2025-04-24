"""
FastAPI wrapper exposing medical-agent over HTTP + SSE
uvicorn -m medical_agent --reload
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from .schemas import ConsultationRequest, ConsultationResponse
from .agents import consult

app = FastAPI(title="Medical AI Agent", version="1.0.0")

@app.post("/v1/consult", response_model=ConsultationResponse)
def consult_endpoint(req: ConsultationRequest):
    try:
        return consult(req)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/healthz")
def healthz():
    return {"status": "ok"} 
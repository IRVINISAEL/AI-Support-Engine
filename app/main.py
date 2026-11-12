from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.traditional import analyze_traditional
from app.generative import analyze_generative, GenerativeError

app = FastAPI(title="AI Support Engine")


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if req.mode == "traditional":
        return analyze_traditional(req.message)
    try:
        return analyze_generative(req.message)
    except GenerativeError as e:
        raise HTTPException(status_code=502, detail=str(e))
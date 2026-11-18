from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.traditional import analyze_traditional
from app.generative import analyze_generative, GenerativeError
from app.ml import analyze_ml, MLError

app = FastAPI(title="AI Support Engine")


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if req.mode == "traditional":
        return analyze_traditional(req.message)
    if req.mode == "ml":
        try:
            return analyze_ml(req.message)
        except MLError as e:
            raise HTTPException(status_code=503, detail=str(e))
    try:
        return analyze_generative(req.message)
    except GenerativeError as e:
        raise HTTPException(status_code=502, detail=str(e))
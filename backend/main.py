from fastapi import FastAPI
from models import AnalyzeRequest

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    return {"status": "received"}
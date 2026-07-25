import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import companies
from app.core.config import PROCESSED_DIR

app = FastAPI(title="기업건강검진 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(companies.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-metrics")
def model_metrics():
    path = PROCESSED_DIR / "pipeline_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

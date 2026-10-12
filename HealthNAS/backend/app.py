from fastapi import FastAPI
from pydantic import BaseModel

from llm.llm_pipeline import HealthcareLLMPipeline


# Create the FastAPI application
app = FastAPI(
    title="Healthcare LLM-Guided NAS",
    description=(
        "LLM + RAG system for "
        "healthcare-specific Neural Architecture Search"
    ),
    version="1.0"
)


# Create the pipeline once when the server starts
pipeline = HealthcareLLMPipeline()


# Request format for the analyze endpoint
class QueryRequest(BaseModel):
    query: str


# Basic API health check
@app.get("/")
def home():
    return {
        "project": (
            "LLM-Guided Neural Architecture Search "
            "for Domain-Specific AI"
        ),
        "domain": "Healthcare",
        "status": "running"
    }


# Send a healthcare ML problem to the LLM + RAG pipeline
@app.post("/llm/analyze")
def analyze(request: QueryRequest):

    result = pipeline.process(
        request.query
    )

    return result
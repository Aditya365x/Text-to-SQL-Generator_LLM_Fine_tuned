from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import csv
import io
from typing import Optional

app = FastAPI(title="SQL Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://adityaXXXXXX-gemma-sql-api.hf.space/generate"

class GenerateRequest(BaseModel):
    question: str
    schema_context: Optional[str] = None

class GenerateResponse(BaseModel):
    question: str
    sql: str
    schema_used: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if req.schema_context:
        enriched_question = f"Given table with schema: {req.schema_context}. {req.question}"
    else:
        enriched_question = req.question

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                HF_API_URL,
                json={"question": enriched_question}
            )
            response.raise_for_status()
            data = response.json()
            return GenerateResponse(
                question=req.question,
                sql=data["sql"],
                schema_used=req.schema_context
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="HF Space timeout — try again")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"HF Space error: {str(e)}")

@app.post("/parse-csv")
async def parse_csv(file_content: str):
    try:
        reader = csv.reader(io.StringIO(file_content))
        headers = next(reader)
        headers = [h.strip().replace(" ", "_").lower() for h in headers]
        return {
            "columns": headers,
            "schema_context": f"table: {', '.join(headers)}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {str(e)}")

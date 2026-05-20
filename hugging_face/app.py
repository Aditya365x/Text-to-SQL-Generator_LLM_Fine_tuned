from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, re

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "google/gemma-3-1b-it"
ADAPTER_ID = "adityaXXXXXX/gemma-1b-sql-final-2000-suffled"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map={"": "cpu"}, token=HF_TOKEN)
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

class Request(BaseModel):
    question: str

def clean_sql(raw: str) -> str:
    sql = raw.split("### SQL:")[-1]
    for c in ["###", "**", "\nLet me", "\nNote", "\nExplanation", "\nThis", "\n*", "```", "Let me know"]:
        sql = sql.split(c)[0]
    lines = []
    for line in sql.strip().splitlines():
        line = line.strip()
        if line:
            lines.append(line)
        if line.endswith(";"):
            break
    sql = " ".join(lines).strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: Request):
    prompt = (
        "You are a SQL query generator. "
        "Output only the SQL query. No explanation, no markdown, no comments.\n\n"
        f"### Question: {req.question}\n### SQL:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"question": req.question, "sql": clean_sql(decoded)}
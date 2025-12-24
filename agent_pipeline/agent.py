from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv(override=True)

app = FastAPI()

SYS_PROMPT = '''You are an expert in the Japanese language and Japanese writing systems.

You will receive two versions of the same text:
1. An OCR-generated text.
2. The ground-truth (correct) text.

Your task is to compare these two texts line by line and perform a detailed analysis.

For each line:
- Indicate whether the OCR result is correct or incorrect.
- When an error occurs, clearly show:
  - the incorrect OCR segment,
  - the correct ground-truth segment,
  - the type of error (e.g. wrong kanji, missing character, extra character, kana confusion, punctuation error),
  - a brief explanation of the likely cause of the OCR error (visual similarity, font issues, segmentation, etc.).

Be precise, systematic, and explicit.  
Do not rewrite the text unnecessarily — focus on comparison and analysis.
'''

# Pydantic model za request body
class OCRAnalysisRequest(BaseModel):
    ocr_text: str
    ground_truth: str

def create_ocr_agent():
    return create_agent(
        model="gpt-4o-mini",
        system_prompt=SYS_PROMPT,  # Koristi pravi system prompt
    )

@app.post("/analyze-ocr")
async def analyze_ocr_endpoint(request: OCRAnalysisRequest):
    """
    Endpoint za analizu OCR teksta.
    
    Body:
    {
        "ocr_text": "OCR rezultat",
        "ground_truth": "Tačan tekst"
    }
    """
    try:
        agent = create_ocr_agent()
        
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"""
OCR TEXT:
{request.ocr_text}

GROUND TRUTH TEXT:
{request.ground_truth}
"""
                }
            ]
        })
        
        analysis = result["messages"][-1].content
        
        return {
            "status": "success",
            "analysis": analysis,
            "ocr_text": request.ocr_text,
            "ground_truth": request.ground_truth
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "OCR Analysis API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Pokretanje: uvicorn main:app --reload
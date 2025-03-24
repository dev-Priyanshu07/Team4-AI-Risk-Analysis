from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define request model
class InputData(BaseModel):
    risk_score: int
    justification: str

def compute_confidence_score(risk_score: int, justification: str) -> float:
    """
    Uses FinBERT to assess how well the justification aligns with the risk score.
    Returns a confidence score between 0 and 10.
    """
    # Format input text
    text = f"Risk Score: {risk_score}. Justification: {justification}"

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs).logits

    # Convert logits to probability using softmax
    probabilities = F.softmax(outputs, dim=-1).squeeze()

    # Take the probability of the 'entailment' class (index 2)
    confidence_score = probabilities[2].item() * 10  # Scale to 0-10

    return round(confidence_score, 2)

@app.post("/analyze")
async def analyze(data: InputData):
    """
    API to calculate confidence score.
    """
    confidence_score = compute_confidence_score(data.risk_score, data.justification)

    return {
        "risk_score": data.risk_score,
        "justification": data.justification,
        "confidence_score": confidence_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

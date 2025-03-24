from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize FastAPI
app = FastAPI()

# Load FinBERT model and tokenizer (pre-trained)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def compute_confidence_score(justification, risk_score):
    """
    Uses FinBERT to assess how well the justification aligns with the risk score.
    Returns a confidence score between 0 and 1.
    """
    inputs = tokenizer(f"Risk Score: {risk_score}. Justification: {justification}", 
                       return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Convert logits to probability using softmax
    probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    
    # Take the probability of the 'entailment' class (assumption: index 2 is entailment)
    confidence_score = probabilities[0][2].item()
    
    return round(confidence_score, 2)*10

@app.post("/analyze")
async def analyze(data: dict):
    justification = data.get("justification", "")
    risk_score = data.get("risk_score", None)
    
    if not justification or risk_score is None:
        return {"error": "Both justification and risk_score are required"}
    
    confidence_score = compute_confidence_score(justification, risk_score)
    
    return {
        "risk_score": risk_score,
        "justification": justification,
        "confidence_score": confidence_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load trained models
finbert_model = AutoModelForSequenceClassification.from_pretrained("models/finbert")
finbert_tokenizer = AutoTokenizer.from_pretrained("models/finbert")
justification_model = AutoModelForSeq2SeqLM.from_pretrained("models/flan_t5")
justification_tokenizer = AutoTokenizer.from_pretrained("models/flan_t5")

def analyze_report(report_text):
    inputs = finbert_tokenizer(report_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        risk_level = torch.argmax(outputs.logits).item()

    risk_category = {0: "Low", 1: "Medium", 2: "High"}[risk_level]

    justification_prompt = f"News Report: {report_text}\nRisk Category: {risk_category}\nExplain why this entity is risky."
    inputs = justification_tokenizer(justification_prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():
        output = justification_model.generate(**inputs, max_length=100)
    
    justification = justification_tokenizer.decode(output[0], skip_special_tokens=True)

    return {"risk_category": risk_category, "justification": justification}

report_text = "XYZ Corp is under investigation for fraud."
print(analyze_report(report_text))

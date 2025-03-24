import json
import os
import time
import random
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
from scripts.layer1 import check_company  # Import report generation function

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
DATA_DIR = "data"
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load Pre-trained FinBERT for Risk Scoring
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# List of Companies for Report Generation
companies = [
    "Tesla", "JP Morgan", "Goldman Sachs", "Citibank", "Binance",
    "Meta", "Google", "Amazon", "Microsoft", "Wells Fargo",
    "Morgan Stanley", "HSBC", "Bank of America", "Alibaba", "Credit Suisse"
]

random.shuffle(companies)  # Shuffle for variation

def assign_risk_label(report_text):
    """
    Uses FinBERT to predict risk label (Low, Medium, High).
    """
    inputs = tokenizer(report_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        risk_level = torch.argmax(outputs.logits).item()
    
    return {0: "Low", 1: "Medium", 2: "High"}[risk_level]

def generate_risk_report(company_name):
    """
    Generates a financial risk report for a given company using `check_company()` from layer1.py.
    """
    logging.info(f"Generating risk report for: {company_name}")

    # Generate full report using layer1.py
    report_text = check_company(company_name)

    # Assign risk label automatically
    risk_label = assign_risk_label(report_text)

    # Store report with label
    data_entry = {
        "company": company_name,
        "report_text": report_text,
        "risk_label": risk_label,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(TRAINING_DATA_FILE, "a") as f:
        f.write(json.dumps(data_entry) + "\n")

    logging.info(f"✅ Report saved for {company_name} - Risk: {risk_label}")

def generate_multiple_reports(num_reports=100):
    """
    Generates multiple risk reports for training data.
    """
    count = 0
    for company in companies:
        try:
            generate_risk_report(company)
            count += 1
            logging.info(f"✅ {count}/{num_reports} reports generated.")
            
            if count >= num_reports:
                break

            time.sleep(random.uniform(1, 3))  # Avoid API rate limits
        except Exception as e:
            logging.error(f"⚠️ Error processing {company}: {e}")

    logging.info(f"\n🎯 Successfully generated {count} reports!")

if __name__ == "__main__":
    generate_multiple_reports(100)

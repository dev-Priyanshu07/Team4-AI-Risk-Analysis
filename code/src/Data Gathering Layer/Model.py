import requests
import pandas as pd
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import pickle
import xml.etree.ElementTree as ET
import logging
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

OFAC_CSV_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
OFAC_FILE = "ofac_sanctions.csv"
OFAC_EMBEDDINGS_FILE = "ofac_embeddings.pkl"

HEADERS = {"User-Agent": "my-sec-bot/1.0 (contact: myemail@example.com)"}

# Load NLP Models
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("text-classification", model=finbert_model, tokenizer=tokenizer)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner_model = spacy.load("en_core_web_sm")


def download_ofac_list():
    if os.path.exists(OFAC_FILE):
        logging.info("OFAC sanctions list already exists. Skipping download.")
        return
    
    response = requests.get(OFAC_CSV_URL, stream=True)
    if response.status_code == 200:
        with open(OFAC_FILE, "wb") as file:
            file.write(response.content)
        logging.info("OFAC sanctions list downloaded successfully.")
    else:
        logging.error("Failed to download OFAC list.")


def clean_name(name):
    return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().lower()


def load_ofac_data():
    try:
        df = pd.read_csv(OFAC_FILE, encoding="ISO-8859-1", header=None)
        df = df[[1]]
        df.columns = ["Name"]
        df["Cleaned Name"] = df["Name"].apply(clean_name)
        return df
    except Exception as e:
        logging.error(f"Error loading OFAC data: {e}")
        return None


def get_sanctioned_embeddings():
    df = load_ofac_data()
    if df is None:
        return None, None
    
    sanctioned_names = df["Cleaned Name"].dropna().tolist()
    
    if os.path.exists(OFAC_EMBEDDINGS_FILE):
        with open(OFAC_EMBEDDINGS_FILE, "rb") as f:
            cached_data = pickle.load(f)
            if cached_data["names"] == sanctioned_names:
                logging.info("Loaded cached embeddings.")
                return df, cached_data["embeddings"]
    
    sanctioned_embeddings = model.encode(sanctioned_names, convert_to_tensor=True)
    
    with open(OFAC_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"names": sanctioned_names, "embeddings": sanctioned_embeddings}, f)
    
    logging.info("Embeddings computed & cached.")
    return df, sanctioned_embeddings

# Function to Perform Sentiment Analysis on Financial News
def get_financial_news(company_name):
    API_KEY = "9047082950c04406ad8378594370e334"  # Replace with your API key
    search_query = f'"{company_name}" AND ("SEC investigation" OR "earnings report" OR "fraud" OR "merger")'
    url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()
    
    if "articles" in data:
        news_list = [
            {"title": article["title"], "description": article["description"]}
            for article in data["articles"][:15]
        ]
        logging.info(f"Fetched {len(news_list)} news articles for {company_name}")
        
        # Filter articles to make sure they actually mention the company
        filtered_news = filter_relevant_news(news_list, company_name)
        return filtered_news
    
    logging.warning(f"No relevant financial news found for {company_name}")
    return []

# Function to Filter News Using Named Entity Recognition (NER)
def filter_relevant_news(news_list, company_name):
    filtered_news = []
    for article in news_list:
        text = f"{article['title']} {article.get('description', '')}"
        doc = ner_model(text)

        # Check if the company name appears in the recognized entities
        if any(ent.text.lower() == company_name.lower() for ent in doc.ents):
            filtered_news.append(article)

    logging.info(f"Filtered news count: {len(filtered_news)}")
    return filtered_news


def analyze_news_sentiment(news_list):
    sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}
    analyzed_news = []

    for article in news_list:
        title = article["title"]
        description = article.get("description", "")
        text = f"{title}. {description}" if description else title

        sentiment = sentiment_analyzer(text[:512])[0]  # Truncate to fit model limit
        sentiment_label = sentiment["label"].lower()
        sentiment_scores[sentiment_label] += 1
        analyzed_news.append(f"- {title} ({sentiment_label.capitalize()})")

    # Determine overall sentiment
    overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    return analyzed_news, overall_sentiment

def check_sanctions(name, bert_threshold=0.75, fuzzy_threshold=85):
    df, sanctioned_embeddings = get_sanctioned_embeddings()
    if df is None:
        return "Sanctions data not available."

    name_cleaned = clean_name(name)
    sanctioned_names = df["Cleaned Name"].dropna().tolist()

    input_embedding = model.encode(name_cleaned, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(input_embedding, sanctioned_embeddings)[0]

    bert_matches = [(df.iloc[i]["Name"], similarity_scores[i].item()) for i in range(len(sanctioned_names)) if similarity_scores[i] >= bert_threshold]
    fuzzy_matches = [(df.iloc[i]["Name"], fuzz.ratio(name_cleaned, sanctioned_names[i])) for i in range(len(sanctioned_names)) if fuzz.ratio(name_cleaned, sanctioned_names[i]) >= fuzzy_threshold]
    
    matches = set(bert_matches + fuzzy_matches)
    if matches:
        return f"{name} is potentially sanctioned: {matches}"
    return f"{name} is NOT on the sanctions list."


def get_cik_number(company_name):
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        cik_data = response.json()
        for entry in cik_data.values():
            if company_name.lower() in entry["title"].lower():
                return str(entry["cik_str"]).zfill(10)
    return None

def get_sec_filings(cik_number):
    if not cik_number:
        return "Invalid CIK number"

    sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_number}&output=atom"
    
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//atom:entry", ns)

        filings = []
        for entry in entries[:5]:  # Get latest 5 filings
            title_elem = entry.find("atom:title", ns)
            link_elem = entry.find("atom:link", ns)

            if title_elem is not None and link_elem is not None:
                title = title_elem.text
                link = link_elem.attrib.get("href", "No Link Available")
                filings.append({"title": title, "link": link})

        return filings if filings else "No filings found."
    else:
        return f"Error: {response.status_code}, {response.text}"

def get_financial_data(cik, concept):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return data.get("units", {}).get("USD", [])
    return None


def compile_risk_data(company_name):
    cik = get_cik_number(company_name)
    if not cik:
        return "CIK not found."
    report = f"Entity: {company_name}\nCIK: {cik}\n"
    report+= get_sec_filings(cik)
    summary = summarizer(report[:1024], max_length=150, min_length=50, do_sample=False)
    report += f"Risk Summary: {summary[0]['summary_text']}"
    return report

def check_company(name):
    sanctions_result = check_sanctions(name)
    risk_result = compile_risk_data(name)
    
    report = f"Sanctions Check:\n{sanctions_result}\n\nFinancial Risk:\n{risk_result}\n\n"
    
    news_list = get_financial_news(company_name)
    if news_list:
        analyzed_news, overall_sentiment = analyze_news_sentiment(news_list)
        report += "\n📰 Financial News Sentiment: " + overall_sentiment.capitalize() + "\n"
        report += "\n".join(analyzed_news) + "\n\n"
    else:
        report += "\n📰 Financial News Sentiment: No relevant news available.\n"
    
    summary_text = summarizer(report[:1024], max_length=150, min_length=50, do_sample=False)
    report += f"Short Summary:\n{summary_text[0]['summary_text']}"
    
    return report


# Example Usage
test_companies = ["Tesla", "Wells Fargo", "Hezbollah"]
for company in test_companies:
    print(check_company(company))

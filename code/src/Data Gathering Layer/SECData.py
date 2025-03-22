import requests
import xml.etree.ElementTree as ET
import logging
import spacy
from transformers import pipeline
from datetime import datetime


# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load NLP Models
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner_model = spacy.load("en_core_web_sm")  # Named Entity Recognition

# SEC API Headers
HEADERS = {"User-Agent": "my-sec-bot/1.0 (contact: myemail@example.com)"}

def get_financial_data(cik, concept):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        if "units" in data and "USD" in data["units"]:
            records = data["units"]["USD"]
            financials = {}
            current_year = datetime.now().year
            
            for entry in records:
                if "end" in entry:
                    year = int(entry["end"][:4])
                    if current_year - 4 <= year <= current_year:
                        financials[year] = entry["val"]
            
            return financials
    return None

# Function to Fetch Financial News
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

# Function to Perform Sentiment Analysis on Financial News
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

# Function to Compile Risk Data
def compile_risk_data(company_name):
    cik = get_cik_number(company_name)
    if not cik:
        return "❌ CIK not found."

    report = f"🔹 Entity: {company_name}\n🔹 CIK: {cik}\n\n"

    # Fetch latest SEC filings
    sec_filings = get_sec_filings(cik)

    if isinstance(sec_filings, list) and all(isinstance(item, dict) for item in sec_filings):
        report += "📄 Latest SEC Filings:\n"
        for filing in sec_filings:
            title = filing.get("title", "Unknown Filing")  # Extract a relevant field
            date = filing.get("date", "N/A")
            report += f"  - {title} ({date})\n"
        report += "\n"
    else:
        report += "📄 Latest SEC Filings:\n❌ No data available.\n\n"


    # Fetch financial statements
    risk_metrics = ["Assets", "Liabilities", "Revenues", "NetIncome", "TotalDebt", "OperatingCashFlow"]
    financial_data = {}
    for metric in risk_metrics:
        financial_data[metric] = get_financial_data(cik, metric)
        if financial_data[metric]:
            print(f"  🔹 {metric}:")
            for year, value in sorted(financial_data[metric].items(), reverse=True):
                print(f"    📅 {year}: ${value:,.2f}")
        else:
            print(f"  ❌ No data found for {metric}")

    # Fetch and analyze financial news
    news_list = get_financial_news(company_name)
    if news_list:
        analyzed_news, overall_sentiment = analyze_news_sentiment(news_list)
        report += "\n📰 Financial News Sentiment: " + overall_sentiment.capitalize() + "\n"
        report += "\n".join(analyzed_news) + "\n\n"
    else:
        report += "\n📰 Financial News Sentiment: No relevant news available.\n"

    # Risk Analysis using AI
    report += "\n⚠️ **Risk Assessment Summary:**\n"
    try:
        risk_input = report[:1024]  # Truncate input to fit model limits
        risk_summary = summarizer(risk_input, max_length=150, min_length=50, do_sample=False)
        report += risk_summary[0]["summary_text"]
    except Exception as e:
        logging.error(f"Failed to generate risk analysis: {str(e)}")
        report += "❌ Could not generate risk analysis."

    return report

# Example Usage
company_name = "Wells Fargo"
final_report = compile_risk_data(company_name)
print("\n📝 Final Risk Report:\n", final_report)

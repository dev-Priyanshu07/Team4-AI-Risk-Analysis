import requests
import xml.etree.ElementTree as ET
import logging
from transformers import pipeline

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load NLP Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")

# SEC API Headers
HEADERS = {"User-Agent": "my-sec-bot/1.0 (contact: myemail@example.com)"}

# Function to Fetch Financial News
def get_financial_news(company_name):
    API_KEY = "9047082950c04406ad8378594370e334"  # Replace with your API key
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "articles" in data:
        news_list = [{"title": article["title"], "description": article["description"]} for article in data["articles"][:10]]
        logging.info(f"Fetched {len(news_list)} news articles for {company_name}")
        return news_list
    logging.warning(f"No financial news found for {company_name}")
    return []

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

# Function to Compile Risk Data
def compile_risk_data(company_name):
    cik = get_cik_number(company_name)
    if not cik:
        return "❌ CIK not found."

    report = f"🔹 Entity: {company_name}\n🔹 CIK: {cik}\n\n"

    # Fetch latest SEC filings
    report += "📄 Latest SEC Filings:\n" + get_sec_filings(cik) + "\n\n"
    
    # Fetch financial statements
    financial_statements = get_financial_statements(cik)
    report += "📊 Full Financial Statements (Past 4 Years):\n"
    if financial_statements:
        for concept, values in financial_statements.items():
            report += f"  🔹 {concept}:\n"
            for record in values:
                if "end" in record:
                    year = record["end"][:4]
                    value = record["val"]
                    report += f"    {year}: ${value:,.2f}\n"
    else:
        report += "  ❌ No financial data found.\n"

    # Fetch and analyze financial news
    news_list = get_financial_news(company_name)
    if news_list:
        analyzed_news, overall_sentiment = analyze_news_sentiment(news_list)
        report += "\n📰 Financial News Sentiment: " + overall_sentiment.capitalize() + "\n"
        report += "\n".join(analyzed_news) + "\n\n"
    else:
        report += "\n📰 Financial News Sentiment: No recent news available.\n"

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

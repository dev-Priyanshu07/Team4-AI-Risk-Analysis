import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from transformers import pipeline

# SEC API Headers
HEADERS = {"User-Agent": "my-sec-bot/1.0 (contact: myemail@example.com)"}

# Function to Get CIK Number from Company Name
def get_cik_number(company_name):
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        cik_data = response.json()
        for entry in cik_data.values():
            if company_name.lower() in entry["title"].lower():
                return str(entry["cik_str"]).zfill(10)
    return None

# Function to Get SEC Filings Using CIK Number
def get_sec_filings(cik_number):
    if not cik_number:
        return "Invalid CIK number"
    sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_number}&output=atom"
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//atom:entry", ns)
        filings = [f"- {entry.find('atom:title', ns).text}: {entry.find('atom:link', ns).attrib.get('href', 'No Link')}" for entry in entries[:5]]
        return "\n".join(filings) if filings else "No filings found."
    return f"Error: {response.status_code}, {response.text}"

# Function to Fetch Full Financial Statements from SEC API
def get_financial_statements(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        statements = {}
        for concept, details in data.get("facts", {}).get("us-gaap", {}).items():
            if "units" in details and "USD" in details["units"]:
                statements[concept] = details["units"]["USD"]
        return statements
    return {}

# Function to Fetch Financial News
def get_financial_news(company_name):
    API_KEY = "9047082950c04406ad8378594370e334"  # Replace with your API key
    cik = get_cik_number(company_name)
    executives = get_executives_from_sec(cik) if cik else []
    keywords = f'"{company_name}" OR "{company_name} stock"'
    if executives:
        keywords += " OR " + " OR ".join([f'"{exec_name}"' for exec_name in executives])
    url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "articles" in data:
        return [f"- {article['title']}: {article['url']}" for article in data["articles"][:15]]
    return []

# Function to Fetch Executives from SEC
def get_executives_from_sec(cik):
    sec_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return [officer["name"] for officer in data.get("officers", [])[:5]]
    return []

# Function to Compile Risk Data
def compile_risk_data(company_name):
    cik = get_cik_number(company_name)
    if not cik:
        return "❌ CIK not found."
    
    report = f"🔹 Entity: {company_name}\n"
    report += f"🔹 CIK: {cik}\n\n"
    
    # Fetch latest SEC filings
    report += "📄 Latest SEC Filings:\n" + get_sec_filings(cik) + "\n\n"
    
    # Fetch financial statements
    financial_statements = get_financial_statements(cik)
    report += "Full Financial Statements (Past 4 Years):\n"
    if financial_statements:
        for concept, values in financial_statements.items():
            report += f"  🔹 {concept}:\n"
            for record in values:
                if "end" in record:
                    year = record["end"][:4]
                    value = record["val"]
                    report += f"{year}: ${value:,.2f}\n"
    else:
        report += "  ❌ No financial data found.\n"
    
    # Fetch financial news
    news_articles = get_financial_news(company_name)
    report += "\nLatest Financial News:\n"
    report += "\n".join(news_articles) if news_articles else "  ❌ No relevant news found."
    
    return report

# Example Usage
company_name = "Tesla"
final_report = compile_risk_data(company_name)
print("\n📝 Final Risk Report:\n", final_report)

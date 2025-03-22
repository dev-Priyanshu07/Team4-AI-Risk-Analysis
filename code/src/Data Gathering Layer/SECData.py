import requests
import xml.etree.ElementTree as ET
from datetime import datetime

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

# Function to Fetch Financial Data from SEC API
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

# Function to Compile All Risk-Related Data
def compile_risk_data(company_name):
    cik = get_cik_number(company_name)
    if not cik:
        print("❌ CIK not found.")
        return
    
    print(f"🔹 CIK for {company_name}: {cik}")
    
    # Fetch latest SEC filings
    filings = get_sec_filings(cik)
    print("\n📄 Latest SEC Filings:")
    if isinstance(filings, list):
        for filing in filings:
            print(f"- {filing['title']}: {filing['link']}")
    else:
        print(filings)
    
    # Fetch financial risk-related data
    risk_metrics = ["Assets", "Liabilities", "Revenues", "NetIncome", "TotalDebt", "OperatingCashFlow"]
    financial_data = {}
    
    print("\n📊 Financial Data (Past 4 Years):")
    for metric in risk_metrics:
        financial_data[metric] = get_financial_data(cik, metric)
        if financial_data[metric]:
            print(f"  🔹 {metric}:")
            for year, value in sorted(financial_data[metric].items(), reverse=True):
                print(f"    📅 {year}: ${value:,.2f}")
        else:
            print(f"  ❌ No data found for {metric}")

# Example Usage
company_name = "Tesla"  # Change this to any public company
compile_risk_data(company_name)

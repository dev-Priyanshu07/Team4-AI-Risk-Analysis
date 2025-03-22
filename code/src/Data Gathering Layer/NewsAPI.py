import requests

def get_financial_news(company_name):
    API_KEY = "9047082950c04406ad8378594370e334"  # Replace with your API key
    url = f"https://newsapi.org/v2/everything?q={company_name} \
            +finance+business+stocks+investment+earnings&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    if "articles" in data:
        return [article["description"] for article in data["articles"]]
    else:
        return []
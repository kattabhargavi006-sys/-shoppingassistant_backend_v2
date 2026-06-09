from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
import re
import requests

load_dotenv()

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "api running successfully"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LLM ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# ---------------- JSON helper ----------------
def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON found")

# ---------------- SIMPLE SEARCH (NO TAVILY) ----------------
def simple_search(query: str):
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)

    return r.text[:3000]   # just raw text for LLM

# ---------------- TOOLS ----------------

@tool
def product_api(product_query: str):
    """Fetch product details and return structured JSON."""
    text = simple_search(f"best {product_query} price rating")

    prompt = f"""
Extract 5 products from this text.

Return ONLY JSON:
{{
  "products": [
    {{"name": "", "price": "", "rating": ""}}
  ]
}}

TEXT:
{text}
"""
    content = llm.invoke(prompt).content

    try:
        return extract_json(content)
    except:
        return {"products": [], "raw": content}


@tool
def price_compare(product: str):
    """Compare product prices."""
    text = simple_search(f"{product} Amazon Flipkart price")

    prompt = f"""
Extract price comparison in JSON:

{{
  "platforms": [
    {{"site": "", "price": "", "discount": ""}}
  ]
}}

TEXT:
{text}
"""
    content = llm.invoke(prompt).content

    try:
        return extract_json(content)
    except:
        return {"platforms": [], "raw": content}


@tool
def review_analyzer(product_name: str):
    """Analyze reviews."""
    text = simple_search(f"{product_name} reviews pros cons")

    prompt = f"""
Extract:
- pros (list)
- cons (list)
- summary (2 sentences)

Return JSON only:
{{
  "pros": [],
  "cons": [],
  "summary": ""
}}

TEXT:
{text}
"""
    content = llm.invoke(prompt).content

    try:
        return extract_json(content)
    except:
        return {"pros": [], "cons": [], "summary": content}


# ---------------- AGENT ----------------
agent = create_react_agent(
    model=llm,
    tools=[product_api, price_compare, review_analyzer]
)

# ---------------- APIs ----------------

@app.post("/get_product")
def get_products(product_query: str = Query(...)):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Search products: {product_query}"}]
    })

    return {"answer": result["messages"][-1].content}


@app.post("/get_price_compare")
def get_price_compare(product: str = Query(...)):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Compare prices: {product}"}]
    })

    return {"answer": result["messages"][-1].content}


@app.post("/analyze_reviews")
def analyze_reviews(product_name: str = Query(...)):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze reviews: {product_name}"}]
    })

    return {"answer": result["messages"][-1].content}
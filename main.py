from fastapi import FastAPI,Query
from langchain_core.tools import tool
from fastapi.middleware.cors import CORSMiddleware
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from tavily.client import TavilyClient
from dotenv import load_dotenv
import os
import json,re
load_dotenv()
import requests
app=FastAPI()
@app.get("/")
def home():
    return{
        "msg":"api running successfully"
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)
client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)
@tool
def product_api(product_query: str):
    """Fetch product details and return structured JSON with name, price, rating."""
    result = client.search(
        query=f"best {product_query} price rating specifications",
        max_results=5
    )
    text = "\n".join(r["content"] for r in result["results"])
    prompt = f"""
You are a product information extractor.
Extract 5 products from the text below.

RULES:
- Always provide a rating between 0 and 5 (estimate if missing)
- Price must be in INR (₹)
- Return ONLY valid JSON, no explanation, no markdown,no footnotes
- Do NOT add next steps, suggestions, or any extra text after the JSON
FORMAT:
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
    except Exception:
        return {"products": [], "error": "Failed to parse products", "raw": content}
def extract_json(text: str):
    """Extract JSON from model output even if wrapped in markdown."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON found in response")
@tool
def price_compare(product: str):
    """Compare prices for a product across Amazon, Flipkart, Croma, Meesho."""
    result = client.search(
        query=f"{product} price Amazon Flipkart Croma Meesho",
        max_results=7
    )
    text = "\n".join(r.get("content", r.get("snippet", "")) for r in result["results"])
    prompt = f"""
Extract price comparison for: {product}

RULES:
- Include site name, price in INR, and discount if available
- Return ONLY valid JSON, no explanation
FORMAT:
{{
  "platforms": [
    {{"site": "Amazon", "price": "₹XX,XXX", "discount": "-10%"}}
  ]
}}
TEXT:
{text}
"""
    content = llm.invoke(prompt).content
    try:
        return extract_json(content)
    except Exception:
        return {"platforms": [], "error": "Failed to parse prices", "raw": content}
@tool
def review_analyzer(product_name: str):
    """Fetch and analyze product reviews, return pros, cons and summary."""
    result = client.search(
        query=f"{product_name} user reviews pros cons",
        max_results=5
    )
    text = "\n".join(r["content"] for r in result["results"])

    prompt = f"""Product: {product_name}

Extract from the reviews below:
- pros: up to 5 positive points as a list
- cons: up to 5 negative points as a list  
- summary: exactly 2 sentences,no source attribution, no footnotes

REVIEWS:
{text}"""

    structured_llm = llm.with_structured_output({
        "title": "ReviewAnalysis",
        "type": "object",
        "properties": {
            "pros": {"type": "array", "items": {"type": "string"}},
            "cons": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"}
        },
        "required": ["pros", "cons", "summary"]
    })

    try:
        return structured_llm.invoke(prompt)
    except Exception:
        # fallback to regular invoke
        content = llm.invoke(prompt).content
        try:
            return extract_json(content)
        except Exception:
            return {"pros": [], "cons": [], "summary": content, "error": None}
agent = create_react_agent(model=llm, tools=[product_api, price_compare,review_analyzer])
@app.post("/get_product")
def get_products(product_query: str = Query(...)):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Search for products: {product_query}"}]
    })
    last_message = result["messages"][-1].content
    try:
        parsed = extract_json(last_message)
        return {"products": parsed.get("products", []), "answer": None, "error": None}
    except Exception:
        return {"products": [], "answer": last_message, "error": None}

@app.post("/get_price_compare")
def get_price_compare(product: str = Query(...)):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Compare prices for: {product}"}]
    })
    last_message = result["messages"][-1].content
    try:
        parsed = extract_json(last_message)
        return {"platforms": parsed.get("platforms", []), "answer": None, "error": None}
    except Exception:
        # Model gave text/table instead of JSON — return as answer, not error
        return {"platforms": [], "answer": last_message, "error": None}
@app.post("/analyze_reviews")
def analyze_reviews(product_name: str = Query(...)):
    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze reviews for: {product_name}"}]
    })
    last_message = result["messages"][-1].content
    try:
        parsed = extract_json(last_message)
        return {
            "pros": parsed.get("pros", []),
            "cons": parsed.get("cons", []),
            "summary": parsed.get("summary", ""),
            "error": None
        }
    except Exception:
        # JSON failed — return raw as answer
        return {"pros": [], "cons": [], "summary": "", "answer": last_message, "error": None}
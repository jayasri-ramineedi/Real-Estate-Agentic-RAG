import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing. Check your .env file.")

if not TAVILY_API_KEY:
    raise ValueError("Tavily API key is missing. Check your .env file.")

print("API keys loaded successfully.")

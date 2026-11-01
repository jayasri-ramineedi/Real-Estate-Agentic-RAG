
from pathlib import Path
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Step 1: Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Step 2: Create the Tavily Search Tool
web_search_tool = TavilySearch(
    max_results=3
)

# Step 3: Search the Web
def search_web(question: str):

    results = web_search_tool.invoke(question)

    formatted_results = []

    for result in results.get("results", []):

        formatted_results.append({
            "title": result.get("title", "Unknown"),
            "url": result.get("url", ""),
            "content": result.get("content", "")
        })

    return formatted_results


# Step 4: Test the Web Search Tool
if __name__ == "__main__":

    question = input("Enter your question: ")

    answer = search_web(question)

    print("\nWeb Search Results:\n")

    for result in answer:
        print("Title:", result["title"])
        print("URL:", result["url"])
        print("Content:", result["content"])
        print("-" * 50)
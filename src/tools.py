from langchain_core.tools import tool

from pdf_tool import search_pdfs
from web_search import search_web

# Step 1: Create PDF Search Tool
@tool
def pdf_search_tool(question: str) -> str:
    """Search the real estate PDF documents for relevant information."""
    return search_pdfs(question)


# Step 2: Create Web Search Tool
@tool
def web_search_tool(question: str) -> str:
    """Search the web and return relevant information with source URLs."""

    results = search_web(question)

    formatted_results = []

    for result in results:
        formatted_results.append(
            f"Title: {result['title']}\n"
            f"Source URL: {result['url']}\n"
            f"Content: {result['content']}"
        )

    return "\n\n".join(formatted_results)


# Step 3: Collect All Tools
tools = [
    pdf_search_tool,
    web_search_tool
]
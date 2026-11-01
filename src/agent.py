
import re

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY
from tools import tools


# Step 1: Initialize Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)


# Step 2: Create the AI Agent
agent = create_agent(
    model=llm,
    tools=tools,

    system_prompt="""
You are a helpful Real Estate Assistant.

Follow these rules carefully:

1. For property-specific questions, use the PDF search tool first.
2. For current real estate trends and general information, use the web search tool.
3. Answer using only the information found in the retrieved documents or web search results.
4. Do not invent prices, rates, dates, measurements, or other facts.
5. Do not calculate or assume missing values unless the user asks for a calculation.
6. If the retrieved information contains conflicting values, clearly mention the inconsistency instead of choosing one.
7. If the required information is not available, clearly say so.
8. Keep answers clear, accurate, and simple.
9. Do not include a separate Sources section or source URLs in the final answer. The application displays reference sources separately.
10. Before using retrieved information, verify that it belongs to the property mentioned in the user's question.

11. Never use information about one property to answer a question about another property.

12. If the retrieved documents contain information about a different property, ignore that information. If the requested property's information is unavailable, clearly say that it was not found.
"""
)


# Step 3: Get Answer, Context and Sources from the Agent
def ask_agent(question: str):

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    answer = response["messages"][-1].text

    sources = []
    contexts = []
    source_types = []

    # Track duplicate chunks
    seen_chunks = set()

    for message in response["messages"]:

        tool_name = getattr(message, "name", None)

        # Capture PDF and Web tool results
        if tool_name in ["pdf_search_tool", "web_search_tool"]:

            content = message.content

            if isinstance(content, str):

                # Split retrieved content into individual chunks
                blocks = content.split("\n\n")

                for block in blocks:

                    block = block.strip()

                    if not block:
                        continue

                    # Normalize whitespace for duplicate checking
                    normalized_block = re.sub(
                        r"\s+",
                        " ",
                        block
                    ).strip()

                    # Add only unique chunks
                    if normalized_block not in seen_chunks:

                        seen_chunks.add(normalized_block)
                        contexts.append(block)

                if tool_name == "pdf_search_tool":
                    source_types.append("PDF")

                else:
                    source_types.append("Web")

            # Extract web source links
            if tool_name == "web_search_tool" and isinstance(content, str):

                for block in content.split("\n\n"):

                    if "Source URL:" in block:

                        lines = block.splitlines()

                        title = lines[0].replace("Title: ", "")
                        url = lines[1].replace("Source URL: ", "")

                        sources.append({
                            "title": title,
                            "url": url
                        })

    # Add numbering to retrieved chunks
    numbered_context = "\n\n".join(
        f"Retrieved Chunk {i}:\n{chunk}"
        for i, chunk in enumerate(contexts, start=1)
    )

    return {
        "answer": answer,
        "sources": sources,
        "context": numbered_context,
        "chunk_count": len(contexts),
        "source_types": list(dict.fromkeys(source_types))
    }


# Step 4: Test the Agent
if __name__ == "__main__":

    question = input("Ask your real estate question: ")

    result = ask_agent(question)

    print("\nFinal Answer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        print(source["title"])
        print(source["url"])

    print("\nRetrieved Context:\n")
    print(result["context"])

    print("\nTotal Unique Chunks:", result["chunk_count"])
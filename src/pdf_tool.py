
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "faiss_index"


# Step 1: Load the Embedding Model and Saved Vector Store
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store


# Step 2: Search Relevant PDF Chunks
def search_pdfs(question: str) -> str:
    print("\nPDF TOOL QUERY:", question)
    vector_store = load_vector_store()

    results = vector_store.max_marginal_relevance_search(
        question,
        k=5,
        fetch_k=10
    )

    if not results:
        return "No relevant information found in the PDFs."

    # Step 3: Remove Duplicate Chunks
    unique_results = []
    seen_chunks = set()

    for doc in results:
        chunk_text = doc.page_content.strip()

        if chunk_text not in seen_chunks:
            seen_chunks.add(chunk_text)
            unique_results.append(doc)

    # Step 4: Format Retrieved Text and Source Information
    formatted_results = []

    for doc in unique_results:
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "Unknown")

        formatted_results.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {doc.page_content}"
        )

    return "\n\n".join(formatted_results)


if __name__ == "__main__":
    question = input("Enter your question: ")

    answer = search_pdfs(question)

    print("\nRetrieved PDF Information:\n")
    print(answer)

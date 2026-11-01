from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"


def build_index():

    # Step 1: Document Loading
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF documents found in the data folder.")

    print("Total pages loaded:", len(documents))

    # Step 2: Text Splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print("Total chunks created:", len(chunks))

    # Step 3: Create Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 4: Create and Save Vector Store
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vector_store.save_local(str(INDEX_DIR))

    print("Total vectors stored:", vector_store.index.ntotal)
    print("FAISS index saved successfully.")


if __name__ == "__main__":
    build_index()
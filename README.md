# 🏠 Real Estate Agentic RAG Assistant

An AI-powered Real Estate Question Answering system built using Agentic RAG, LangChain, and Google Gemini.

# 📌 Project Overview

This application allows users to ask questions about real estate documents and current real estate market information.

The system uses a LangChain Agent to decide whether to search the available PDF documents or the web based on the user's question.

It uses Hugging Face embeddings, FAISS vector search, Tavily for web search, and Google Gemini for answer generation and evaluation.

# 🚀 Features

- 📄 PDF document processing
- ✂️ Text chunking
- 🔢 Hugging Face text embeddings
- 🔎 Semantic search using FAISS
- 🤖 Agentic AI using LangChain
- 🛠️ PDF Search and Web Search tools
- 🌐 Real-time web search using Tavily
- 💬 Google Gemini for answer generation
- 📊 AI-based answer evaluation
- 🖥️ Streamlit web interface

# 🛠️ Technologies Used

- Python
- LangChain
- Streamlit
- Google Gemini API
- Hugging Face Sentence Transformers
- FAISS
- Tavily Search API
- PyPDF
- python-dotenv

# 🔄 Agentic RAG Workflow

User Question  
↓  
Streamlit Interface  
↓  
LangChain AI Agent  
↓  
Agent Selects the Appropriate Tool  
↓  
PDF Search Tool / Web Search Tool  
↓  
FAISS Vector Search / Tavily Web Search  
↓  
Retrieved Relevant Information  
↓  
Google Gemini  
↓  
Final Answer  
↓  
Gemini-based Answer Evaluation  
↓  
Display Answer and Evaluation in Streamlit

# 📊 Answer Evaluation

The generated answers are evaluated using:

- Faithfulness
- Answer Relevance
- Context Relevance

Each metric is scored on a scale of 1 to 5 using Gemini as an AI evaluator.

# 📂 Project Structure

```text
Real-Estate-Agentic-RAG/
│
├── .gitignore
├── requirements.txt
├── README.md
├── .env
│
├── data/
│   ├── 01_SkyMeadows_Price_List.pdf
│   ├── 02_EmeraldEnclave_Handbook.pdf
│   ├── 03_Listing_Portfolio.pdf
│   └── 04_Maintenance_Compliance_Log.pdf
│
├── faiss_index/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── build_index.py
    ├── pdf_tool.py
    ├── web_search.py
    ├── tools.py
    ├── agent.py
    ├── evaluation.py
    └── app.py
```

# ⚙️ Installation and Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root directory and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Build the FAISS Index

```bash
python src/build_index.py
```

### 6. Run the Streamlit Application

```bash
streamlit run src/app.py
```

# 👩‍💻 Author

**Jayasri Ramineedi**

GitHub: [jayasri-ramineedi](https://github.com/jayasri-ramineedi)

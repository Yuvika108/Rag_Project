# RAG Project

A collection of Retrieval-Augmented Generation (RAG) experiments built with [LangChain](https://www.langchain.com/), Mistral AI, and local vector stores. The centerpiece is a Streamlit app that lets you upload a PDF and ask questions about it. The rest of the repo is a set of standalone scripts exploring different pieces of a RAG pipeline: document loaders, retrievers (similarity, MMR, multi-query), and vector stores (Chroma, FAISS).

## Features

- **📖 RAG Book Assistant** (`app.py`) — Upload a PDF, build a vector index, and ask natural-language questions answered strictly from the document's content.
- **Document loaders** — Examples for loading PDFs (`document_loaders/pdf.py`), plain text (`document_loaders/test.py`), and web pages (`document_loaders/page.py`).
- **Vector stores** — A reusable FAISS + HuggingFace-embeddings module (`vector_store/DB.py`) plus an in-progress Chroma builder (`vector_store/create_database.py`).
- **Retrievers** — Standalone demos of similarity vs. MMR search (`retrievers/mmr.py`), multi-query retrieval (`retrievers/multiquery.py`), and an Arxiv paper retriever (`retrievers/arixv.py`).
- **Summarizer** (`main.py`) — Summarizes `document_loaders/notes.txt` using Mistral, with a local fallback if no API key is set.

## Project Structure

```
Rag_Project/
├── app.py                     # Streamlit RAG chat app
├── main.py                    # Standalone text summarizer demo
├── requirements.txt
├── document_loaders/
│   ├── page.py                 # Load a web page
│   ├── pdf.py                  # Load + chunk a PDF
│   ├── test.py                 # Load + chunk a text file
│   ├── notes.txt                # Sample text used by main.py / test.py
│   ├── GRU.pdf                  # Sample PDF
│   └── deeplearning.pdf         # Sample PDF
├── retrievers/
│   ├── mmr.py                   # Similarity vs. MMR retrieval demo
│   ├── multiquery.py            # Multi-query retriever demo
│   └── arixv.py                 # Arxiv paper retriever demo
├── vector_store/
│   ├── DB.py                    # Build/load a FAISS vector store
│   ├── create_database.py       # (WIP) Chroma database builder
│   └── __init__.py
└── tests/
    ├── conftest.py
    ├── test_main.py
    └── test_vector_store.py
```

## Prerequisites

- Python 3.10+
- A [Mistral AI](https://console.mistral.ai/) API key (used for chat/completion)
- Internet access on first run to download the HuggingFace sentence-transformer embedding model

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/Yuvika108/Rag_Project.git
   cd Rag_Project
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Usage

### Run the RAG Book Assistant

```bash
streamlit run app.py
```

Then, in the browser tab that opens:
1. Upload a PDF.
2. Click **Create Vector Database** to chunk and embed the document.
3. Type a question in the text box — answers are generated only from the uploaded document's content.

### Run the individual demo scripts

```bash
python main.py                      # Summarize document_loaders/notes.txt
python document_loaders/pdf.py      # Load and chunk GRU.pdf
python document_loaders/test.py     # Load and chunk notes.txt
python document_loaders/page.py     # Scrape a sample web page
python vector_store/DB.py           # Build and query a FAISS index
python retrievers/mmr.py            # Compare similarity vs. MMR retrieval
python retrievers/multiquery.py     # Run multi-query retrieval
python retrievers/arixv.py          # Search Arxiv papers
```

### Run the tests

```bash
pytest
```

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Mistral AI (`langchain-mistralai`) |
| Embeddings | HuggingFace Sentence Transformers |
| Vector stores | Chroma, FAISS |
| Orchestration | LangChain |
| UI | Streamlit |
| PDF parsing | pypdf |

## Known Limitations

- `vector_store/create_database.py` is a work in progress and not yet implemented.
- Model versions and embedding providers vary slightly across scripts since each one was built as an independent experiment rather than a unified pipeline.
- The Streamlit app currently re-embeds and reloads the vector store on every interaction; this works but isn't optimized for large documents.

## License

No license specified yet. Add a `LICENSE` file if you intend to open-source this project.

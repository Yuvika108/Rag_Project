from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_dir = Path(__file__).resolve().parent
pdf_path = base_dir / "GRU.pdf"

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found at: {pdf_path}")

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = splitter.split_documents(docs)

if chunks:
    print(chunks[0].page_content)
else:
    print("No chunks were generated from the PDF.")
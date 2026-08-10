"""Build a persistent Chroma vector database from a PDF file."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_chroma_database(
    pdf_path: str | Path,
    persist_directory: str | Path = DEFAULT_CHROMA_DIR,
) -> Chroma:
    """Load a PDF, split it into chunks, and persist it in Chroma."""
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise FileNotFoundError(f"PDF not found: {source_path}")

    loader = PyPDFLoader(str(source_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a persistent Chroma DB from a PDF."
    )
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument(
        "--persist-directory",
        default=DEFAULT_CHROMA_DIR,
        help=f"Directory for the Chroma DB. Defaults to {DEFAULT_CHROMA_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_chroma_database(args.pdf_path, args.persist_directory)
    print(f"Chroma database saved to: {args.persist_directory}")


if __name__ == "__main__":
    main()

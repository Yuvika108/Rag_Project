"""Text loading and chunking demo using LangChain."""

import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

print("Imports working!")

def main():
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=1)

    # Resolve notes.txt relative to this script's directory
    notes_path = Path(__file__).parent / "notes.txt"

    if not notes_path.exists():
        sys.exit(f"notes.txt not found at: {notes_path}")

    loader = TextLoader(str(notes_path))
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        print(chunk.page_content)
        print()


if __name__ == "__main__":
    main()
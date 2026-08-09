import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DEFAULT_DOCS = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"},
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"},
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"},
    ),
]


def build_vectorstore(
    docs: List[Document] | None = None,
    index_dir: str | os.PathLike[str] | None = None,
) -> Tuple[FAISS, str]:
    """Build and persist a local FAISS vector store using sentence-transformers."""
    documents = docs or DEFAULT_DOCS
    target_dir = Path(index_dir or "vector_store/faiss-index")
    target_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(str(target_dir))
    return vectorstore, str(target_dir)


def load_vectorstore(index_dir: str | os.PathLike[str] | None = None) -> FAISS:
    """Load a persisted FAISS vector store from disk."""
    target_dir = Path(index_dir or "vector_store/faiss-index")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(str(target_dir), embeddings, allow_dangerous_deserialization=True)


def main() -> None:
    vectorstore, saved_path = build_vectorstore()
    print(f"Saved FAISS index to: {saved_path}")

    results = vectorstore.similarity_search("What is used for Data Analysis?", k=2)
    for item in results:
        print(item.page_content)
        print(item.metadata)
        print()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("Explain Deep Learning")
    for item in docs:
        print(item.page_content)
        print(item.metadata)
        print()


if __name__ == "__main__":
    main()

"""Summarize notes.txt using Mistral AI via LangChain."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from fastapi import FastAPI

app = FastAPI()  # Must be named "app"

@app.get("/")
def read_root():
  return {"message": "Hello World"}

def create_app():
  fastapi_app = FastAPI()
  # routes and setup
  return fastapi_app


app = create_app()  # Expose global 'app' variable for Vercel

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

def find_notes_file() -> Path:
    """Locate notes.txt in the document loaders directory."""
    base = Path(__file__).parent
    candidate_dirs = [
        base / "document_loaders",
        base / "documents loaders",
        base / "documents-loaders",
    ]

    for directory in candidate_dirs:
        notes_path = directory / "notes.txt"
        if notes_path.exists():
            return notes_path

    raise FileNotFoundError(
        f"notes.txt not found in any expected directory under: {base}"
    )


def summarize_with_mistral(text: str) -> str | None:
    """Attempt to summarize text using the Mistral API."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None

    try:
        model = ChatMistralAI(model="mistral-small-latest", api_key=api_key)
        result = model.invoke(f"Summarize the following text:\n\n{text}")

        if hasattr(result, "content"):
            return result.content
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        return str(result)
    except Exception:
        return None


def summarize_locally(text: str) -> str:
    """Simple local fallback: return the first two non-empty lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
        return ". ".join(parts[:2]) + "." if parts else ""

    return " ".join(lines[:2])


def main():
    notes_path = find_notes_file()
    text = notes_path.read_text(encoding="utf-8").strip()

    summary = summarize_with_mistral(text) or summarize_locally(text)

    print("--- Source text ---")
    print(text)
    print("\n--- Summary ---")
    print(summary)


if __name__ == "__main__":
    main()

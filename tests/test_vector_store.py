import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import vector_store.DB as db_module


def test_build_vectorstore_creates_index(tmp_path):
    index_dir = tmp_path / "faiss-index"
    vectorstore, saved_path = db_module.build_vectorstore(index_dir=index_dir)

    assert saved_path == str(index_dir)
    assert index_dir.exists()

    results = vectorstore.similarity_search("data analysis", k=1)
    assert results
    assert "Pandas" in results[0].page_content

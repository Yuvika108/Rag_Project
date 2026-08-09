from pathlib import Path

import main


def test_find_notes_file_resolves_existing_notes():
    notes_path = main.find_notes_file()

    expected_path = Path(__file__).resolve().parents[1] / "document_loaders" / "notes.txt"

    assert notes_path == expected_path
    assert notes_path.exists()

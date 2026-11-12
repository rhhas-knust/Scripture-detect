import pandas as pd
import pytest
from src.detector import ScriptureDetector

@pytest.fixture
def detector():
    verses = pd.DataFrame([
        {"verse_id": "ROM.8.28", "book_id": "ROM", "book": "Romans", "chapter": 8, "verse": 28, "text": "And we know..."},
        {"verse_id": "JHN.3.16", "book_id": "JHN", "book": "John", "chapter": 3, "verse": 16, "text": "For God so loved..."},
        {"verse_id": "1JN.4.8", "book_id": "1JN", "book": "1 John", "chapter": 4, "verse": 8, "text": "God is love."}
    ])
    return ScriptureDetector(verses)

@pytest.mark.parametrize("text,expected", [
    ("Turn with me to Romans 8:28.", "ROM.8.28"),
    ("John 3:16 is the golden text", "JHN.3.16"),
    ("First John 4:8 says God is love", "1JN.4.8"),
    ("1 john 4.8", "1JN.4.8"),
])
def test_reference_parsing(detector, text, expected):
    hits = detector.detect(text)
    assert any(h.verse_id == expected for h in hits)
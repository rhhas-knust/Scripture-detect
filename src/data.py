from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

BOOKS = [
    ("GEN", "Genesis", ["gen", "gn", "genesis"]), ("EXO", "Exodus", ["exo", "exod", "exodus"]),
    ("LEV", "Leviticus", ["lev", "levi", "leviticus"]), ("NUM", "Numbers", ["num", "numbers"]),
    ("DEU", "Deuteronomy", ["deu", "deut", "deuteronomy"]), ("JOS", "Joshua", ["jos", "josh", "joshua"]),
    ("JDG", "Judges", ["jdg", "judg", "judges"]), ("RUT", "Ruth", ["rut", "ruth"]),
    ("1SA", "1 Samuel", ["1sa", "1sam", "1samuel"]), ("2SA", "2 Samuel", ["2sa", "2sam", "2samuel"]),
    ("1KI", "1 Kings", ["1ki", "1kgs", "1king", "1kings"]), ("2KI", "2 Kings", ["2ki", "2kgs", "2king", "2kings"]),
    ("1CH", "1 Chronicles", ["1ch", "1chr", "1chron", "1chronicles"]), ("2CH", "2 Chronicles", ["2ch", "2chr", "2chron", "2chronicles"]),
    ("EZR", "Ezra", ["ezr", "ezra"]), ("NEH", "Nehemiah", ["neh", "nehemiah"]), ("EST", "Esther", ["est", "esth", "esther"]),
    ("JOB", "Job", ["job"]), ("PSA", "Psalms", ["psa", "ps", "pss", "psalm", "psalms"]), ("PRO", "Proverbs", ["pro", "prov", "proverb", "proverbs"]),
    ("ECC", "Ecclesiastes", ["ecc", "eccl", "ecclesiastes"]), ("SNG", "Song of Solomon", ["sng", "song", "sos", "canticles", "songofsolomon"]),
    ("ISA", "Isaiah", ["isa", "isaiah"]), ("JER", "Jeremiah", ["jer", "jeremiah"]), ("LAM", "Lamentations", ["lam", "lamentations"]),
    ("EZK", "Ezekiel", ["eze", "ezek", "ezekiel"]), ("DAN", "Daniel", ["dan", "daniel"]), ("HOS", "Hosea", ["hos", "hosea"]),
    ("JOL", "Joel", ["joe", "joel"]), ("AMO", "Amos", ["amo", "amos"]), ("OBA", "Obadiah", ["oba", "obad", "obadiah"]),
    ("JON", "Jonah", ["jon", "jonah"]), ("MIC", "Micah", ["mic", "micah"]), ("NAM", "Nahum", ["nah", "nahum"]),
    ("HAB", "Habakkuk", ["hab", "habakkuk"]), ("ZEP", "Zephaniah", ["zep", "zeph", "zephaniah"]), ("HAG", "Haggai", ["hag", "haggai"]),
    ("ZEC", "Zechariah", ["zec", "zech", "zechariah"]), ("MAL", "Malachi", ["mal", "malachi"]), ("MAT", "Matthew", ["mat", "matt", "matthew"]),
    ("MRK", "Mark", ["mrk", "mk", "mark"]), ("LUK", "Luke", ["luk", "lk", "luke"]), ("JHN", "John", ["jhn", "jn", "joh", "john"]),
    ("ACT", "Acts", ["act", "acts"]), ("ROM", "Romans", ["rom", "romans"]), ("1CO", "1 Corinthians", ["1co", "1cor", "1corinthians"]),
    ("2CO", "2 Corinthians", ["2co", "2cor", "2corinthians"]), ("GAL", "Galatians", ["gal", "galatians"]), ("EPH", "Ephesians", ["eph", "ephesians"]),
    ("PHP", "Philippians", ["php", "phil", "philippians"]), ("COL", "Colossians", ["col", "colossians"]), ("1TH", "1 Thessalonians", ["1th", "1thess", "1thessalonians"]),
    ("2TH", "2 Thessalonians", ["2th", "2thess", "2thessalonians"]), ("1TI", "1 Timothy", ["1ti", "1tim", "1timothy"]), ("2TI", "2 Timothy", ["2ti", "2tim", "2timothy"]),
    ("TIT", "Titus", ["tit", "titus"]), ("PHM", "Philemon", ["phm", "phlm", "philemon"]), ("HEB", "Hebrews", ["heb", "hebrews"]),
    ("JAS", "James", ["jas", "jam", "james"]), ("1PE", "1 Peter", ["1pe", "1pet", "1peter"]), ("2PE", "2 Peter", ["2pe", "2pet", "2peter"]),
    ("1JN", "1 John", ["1jn", "1joh", "1john"]), ("2JN", "2 John", ["2jn", "2joh", "2john"]), ("3JN", "3 John", ["3jn", "3joh", "3john"]),
    ("JUD", "Jude", ["jud", "jude"]), ("REV", "Revelation", ["rev", "revelation", "revelations"])
]

ALIAS_TO_ID = {a: bid for bid, _, aliases in BOOKS for a in aliases}
ID_TO_NAME = {bid: name for bid, name, _ in BOOKS}
KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

def download_kjv(dest: Path = RAW_DIR / "en_kjv.json") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    r = requests.get(KJV_URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest

def _match_book_id(name: str, index: int) -> str:
    key = name.lower().replace(" ", "").strip()
    for bid, display, _ in BOOKS:
        if key == display.lower().replace(" ", ""):
            return bid
    return BOOKS[index][0]

def load_verses(raw_path: Path = None) -> pd.DataFrame:
    raw_path = raw_path or download_kjv()
    data = json.loads(Path(raw_path).read_text(encoding="utf-8-sig"))
    rows = []
    for book_idx, book in enumerate(data):
        bid = _match_book_id(book.get("name", ""), book_idx)
        display = ID_TO_NAME[bid]
        for ch_idx, chapter in enumerate(book["chapters"], start=1):
            for v_idx, text in enumerate(chapter, start=1):
                rows.append({
                    "verse_id": f"{bid}.{ch_idx}.{v_idx}", "book_id": bid, "book": display,
                    "chapter": ch_idx, "verse": v_idx, "text": " ".join(str(text).split())
                })
    return pd.DataFrame(rows)

def build_verse_table(out_path: Path = PROCESSED_DIR / "verses.parquet") -> pd.DataFrame:
    df = load_verses()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
PROCESSED_DIR = Path("data/processed")

def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def build_index():
    verses = pd.read_parquet(PROCESSED_DIR / "verses.parquet")
    model = load_model()
    
    print("Encoding verses (this takes ~1-2 mins on GPU)...")
    emb = model.encode(
        verses["text"].tolist(), batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float16)
    
    np.save(PROCESSED_DIR / "embeddings.npy", emb)
    (PROCESSED_DIR / "embedding_model.txt").write_text(DEFAULT_MODEL)
    return verses, emb
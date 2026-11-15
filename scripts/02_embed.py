# scripts/02_embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("allenai/specter2_base")  # 768, cosine

df = pd.read_parquet("data/arxiv_subset.parquet")

# Формат, на якому навчена модель: title [SEP] abstract
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/embeddings.npy", embeddings)

first_norm = np.linalg.norm(embeddings[0])

print(f"Загальна кількість оброблених текстів: {embeddings.shape[0]}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {first_norm:.6f}")

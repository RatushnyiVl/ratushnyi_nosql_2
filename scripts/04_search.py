# scripts/04_search.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
NAMESPACE = "science-papers"
MODEL_NAME = "allenai/specter2_base"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
TOP_K = 5

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

# 1. Підключення до індексу та завантаження моделі
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# 2. Кодування запиту
def encode_query(query:str) -> np.ndarray:
    return model.encode(query, normalize_embeddings=True)


def search(query:str, top_k:int=TOP_K, metadata_filter:dict | None = None):
    results = index.query(
        vector=encode_query(query).tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=metadata_filter,
    )
    hits = []
    for m in results.matches:
        md = m.metadata
        hits.append({
            "id": m.id,
            "score": round(m.score, 4),
            "title": md.get("title", ""),
            "category": md.get("category", ""),
            "year": md.get("year", 0),
            "abstract": md.get("abstract", "")[:150] + "...",
        })
    return pd.DataFrame(hits)


def show(title:str, result:pd.DataFrame):
    print(f"\n=== {title} (знайдено: {len(result)}) ===")
    print(result.to_string(index=False) if len(result) else "Порожній результат")


# 3. Чистий семантичний пошук
query = "teaching machines to recognize objects in pictures"
pinecone_hits = search(query)
show(f'Семантичний пошук: "{query}"', pinecone_hits)

# 4. Пошук із фільтрацією
query2 = "reinforcement learning"
current_year = datetime.now().year

filter_A = {"year": {"$gte": current_year - 5}, "category": {"$eq": "cs.LG"}}
filter_B = {"year": {"$lte": 2015}}

show(f'"{query2}" без фільтра (базова лінія)', search(query2))
show(f"Фільтр A: year >= {current_year - 5} і category = cs.LG", search(query2, metadata_filter=filter_A))
show("Фільтр B: year <= 2015", search(query2, metadata_filter=filter_B))

# 5. Порівняння метрик на локальних ембеддингах
embeddings = np.load(INPUT_EMBEDDINGS)
q = encode_query(query)

dot_scores = embeddings @ q
cos_scores = dot_scores / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q))
l2_scores = np.linalg.norm(embeddings - q, axis=1)

top_dot = np.argsort(-dot_scores)[:TOP_K]
top_cos = np.argsort(-cos_scores)[:TOP_K]
top_l2 = np.argsort(l2_scores)[:TOP_K]

for name, idxs, scores in [
    ("Cosine similarity", top_cos, cos_scores),
    ("Dot product", top_dot, dot_scores),
    ("L2 distance", top_l2, l2_scores),
]:
    print(f"\nTOP {TOP_K} ({name})")
    for rank, i in enumerate(idxs, start=1):
        print(f"{rank}. [{scores[i]:.4f}] {df.iloc[i]['title']}")

# Порівняння
print("\nЗбіг топ-5 (cosine == dot == L2):", list(top_cos) == list(top_dot) == list(top_l2))

pinecone_idx = [int(x.split("_")[1]) for x in pinecone_hits["id"]]
print("Збіг топ-5 Pinecone з локальним cosine:", pinecone_idx == list(top_cos))

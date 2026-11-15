# scripts/05_chunking.py
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
NAMESPACE = "science-chunks"
BATCH_SIZE = 100
N_PAPERS = 30
CHUNK_WORDS = 60      # розмір чанка в словах
OVERLAP_WORDS = 10    # перекриття для fixed-size стратегії
TOP_K = 5

INDEXES = {
    "fixed": "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic",
}

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# 1. 30 статей з найдовшими анотаціями
df["abstract_words"] = df["abstract"].str.split().str.len()
top_papers = df.nlargest(N_PAPERS, "abstract_words")
print(f"Обрано {len(top_papers)} статей; слів в анотаціях: "
      f"{top_papers['abstract_words'].min()}–{top_papers['abstract_words'].max()}")


# 2. Стратегії чанкінгу
def fixed_size_chunks(text: str, chunk_size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Фіксована кількість слів, між сусідніми чанками перекриття overlap слів."""
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[$\\])")


def semantic_chunks(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Об'єднує цілі речення, доки не буде перевищено max_words"""
    chunks, current, current_len = [], [], 0
    for sentence in SENTENCE_SPLIT.split(text):
        n_words = len(sentence.split())
        if current and current_len + n_words > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


STRATEGIES = {"fixed": fixed_size_chunks, "semantic": semantic_chunks}


# 3. Створення індексів
def get_or_create_index(name: str):
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
        print(f"Індекс '{name}' створено")
    return pc.Index(name)


# 4-5. Чанкінг, ембеддинги, завантаження
def build_records(strategy: str) -> list[dict]:
    chunk_fn = STRATEGIES[strategy]
    records = []
    for idx, row in top_papers.iterrows():
        for n, text in enumerate(chunk_fn(row["abstract"])):
            records.append({
                "id": f"paper_{idx}_chunk_{n}",
                "text": text,
                "metadata": {
                    "arxiv_id": row["id"],
                    "title": row["title"],
                    "text": text,
                    "chunk_index": n,
                    "year": int(row["year"]),
                    "category": row["category"],
                },
            })
    return records


indexes = {}
for strategy, index_name in INDEXES.items():
    index = get_or_create_index(index_name)
    indexes[strategy] = index

    records = build_records(strategy)
    avg_words = sum(len(r["text"].split()) for r in records) / len(records)
    print(f"\n[{strategy}] чанків: {len(records)}, середня довжина: {avg_words:.1f} слів")

    embeddings = model.encode(
        [r["text"] for r in records],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Завантаження {index_name}"):
        batch = [
            {"id": r["id"], "values": emb.tolist(), "metadata": r["metadata"]}
            for r, emb in zip(records[start:start + BATCH_SIZE], embeddings[start:start + BATCH_SIZE])
        ]
        index.upsert(vectors=batch, namespace=NAMESPACE)

    # статистика Pinecone оновлюється із затримкою, тож трохи чекаємо
    for _ in range(15):
        ns = index.describe_index_stats().namespaces.get(NAMESPACE)
        if ns and ns.vector_count >= len(records):
            break
        time.sleep(2)


# 6. Пошук по чанках
def search_chunks(query:str, strategy:str, top_k:int=TOP_K):
    vector = model.encode(query, normalize_embeddings=True).tolist()
    res = indexes[strategy].query(
        vector=vector, 
        top_k=top_k, 
        include_metadata=True, 
        namespace=NAMESPACE
    )

    return pd.DataFrame([
        {
            "score": round(m.score, 4),
            "id": m.id,
            "title": m.metadata["title"][:60],
            "text": m.metadata["text"][:120] + "...",
        }
        for m in res.matches
    ])


print("\nОбрані статті:")
for t in top_papers["title"]:
    print(" -", t)

# Підлаштуй запити під теми обраних статей (список вище)
test_queries = [
    "neutron stars and their properties",
    "3D calculations of objects in images",
    "publication of uranium and thorium in Earth's crust and mantle",
]

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 130)

for q in test_queries:
    for strategy in INDEXES:
        print(f'\n=== "{q}" | {strategy} ===')
        print(search_chunks(q, strategy).to_string(index=False))

# scripts/03_load_to_pinecone.py
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
NAMESPACE = "science-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

# 1. Клієнт та індекс
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Free plan
    )
    print("Очікування ініціалізації індексу на сервері Pinecone...")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("Індекс успішно створено та активовано!")

index = pc.Index(INDEX_NAME)

# 2. Завантаження даних
documents = pd.read_parquet(INPUT_PARQUET).to_dict(orient="records")
embeddings = np.load(INPUT_EMBEDDINGS)
assert len(documents) == len(embeddings), "Кількість документів і ембеддингів не збігається"


# 3. Формування записів
def build_vector(i:int, doc:dict, emb:np.ndarray) -> dict:
    return {
        "id": f"paper_{i}",
        "values": emb.tolist(),
        "metadata": {
            "arxiv_id": doc["id"],
            "title": doc["title"],
            "abstract": doc["abstract"][:500],
            "authors": doc["authors"][:200] if doc["authors"] else "Unknown",
            "year": int(doc["year"]) if doc["year"] else 0,
            "category": doc["category"],
        },
    }


# 4. Батчове завантаження з прогресом
total = len(documents)
for start in tqdm(range(0, total, BATCH_SIZE), desc="Завантаження в Pinecone"):
    end = min(start + BATCH_SIZE, total)
    batch = [build_vector(i, documents[i], embeddings[i]) for i in range(start, end)]
    index.upsert(vectors=batch, namespace=NAMESPACE)

# 5. Підсумок (статистика оновлюється із затримкою, тому трохи чекаємо)
count = 0
for _ in range(15):
    stats = index.describe_index_stats()
    ns = stats.namespaces.get(NAMESPACE)
    count = ns.vector_count if ns else 0
    if count >= total:
        break
    time.sleep(2)

print(f"\nВекторів у namespace '{NAMESPACE}': {count}")
print(f"Загальна кількість векторів в індексі '{INDEX_NAME}': {stats.total_vector_count}")
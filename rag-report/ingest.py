import hashlib, pathlib, re, uuid
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff

OLLAMA = "http://localhost:11434"
COLLECTION = "regulations"
DIM = 1024

def load_docs(folder="sample_docs"):
    for p in sorted(pathlib.Path(folder).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        yield p.name, text, hashlib.sha256(text.encode()).hexdigest()

def chunk(text, max_words=120, overlap=20):
    sections = re.split(r"\n(?=## )", text)
    for sec in sections:
        words = sec.split()
        step = max_words - overlap
        for i in range(0, max(len(words), 1), step):
            piece = " ".join(words[i:i + max_words])
            if piece:
                yield piece

def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": "bge-m3", "input": texts})
    r.raise_for_status()
    return r.json()["embeddings"]

def main():
    qd = QdrantClient(url="http://localhost:6333")
    if not qd.collection_exists(COLLECTION):
        qd.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )
    total = 0
    for name, text, digest in load_docs():
        chunks = list(chunk(text))
        vectors = embed(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}:{i}")),
                vector=v,
                payload={"doc": name, "sha256": digest, "chunk": i, "text": c},
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        qd.upsert(COLLECTION, points)
        total += len(points)
        print(f"{name}: {len(points)} чанков")
    print("Всего чанков:", total)

if __name__ == "__main__":
    main()

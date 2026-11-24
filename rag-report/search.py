import sys, requests
from qdrant_client import QdrantClient

OLLAMA = "http://localhost:11434"
q = " ".join(sys.argv[1:]) or "Сколько дней отпуска положено?"

qv = requests.post(f"{OLLAMA}/api/embed", json={"model": "bge-m3", "input": [q]}).json()["embeddings"][0]
hits = QdrantClient(url="http://localhost:6333").query_points("regulations", query=qv, limit=3).points
context = "\n\n".join(f"[{h.payload['doc']}] {h.payload['text']}" for h in hits)

prompt = f"Ответь только по контексту. Если ответа нет, скажи об этом. Укажи документ.\n\nКонтекст:\n{context}\n\nВопрос: {q}"
ans = requests.post(f"{OLLAMA}/api/generate", json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False}).json()["response"]
print("Найдено в:", [h.payload["doc"] for h in hits])
print("Ответ:", ans)

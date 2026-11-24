import requests
from qdrant_client import QdrantClient

GOLDEN = [
    ("За сколько минут до открытия приходит бариста?", "01_smena.md"),
    ("В течение какого времени можно вернуть деньги за напиток?", "02_kassa_vozvrat.md"),
    ("Сколько секунд нужно мыть руки перед сменой?", "03_sanitariya.md"),
    ("Сколько стоит безлактозное молоко?", "04_menu_allergeny.md"),
    ("Какая скидка у студентов?", "05_skidki_loyalnost.md"),
]
qd = QdrantClient(url="http://localhost:6333", check_compatibility=False)
ok = 0
for q, doc in GOLDEN:
    v = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": [q]}).json()["embeddings"][0]
    found = [h.payload["doc"] for h in qd.query_points("regulations", query=v, limit=3).points]
    ok += doc in found
    print(("OK  " if doc in found else "MISS"), q, "->", found)
print(f"recall@3 = {ok}/{len(GOLDEN)}")

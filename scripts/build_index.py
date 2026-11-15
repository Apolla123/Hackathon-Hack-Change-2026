"""Заранее считает эмбеддинги фрагментов базы знаний (BAAI/bge-m3) и сохраняет в data/index/embeddings.json.

    python scripts/build_index.py

Нужны EMBED_API_URL/EMBED_API_KEY (или LLM_API_URL/LLM_API_KEY). Готовый индекс можно включить в решение:
тогда при запуске эмбеддинги документов не считаются, а в рантайме запрашивается только эмбеддинг вопроса.
Без индекса бот работает на BM25.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import llm  # noqa: E402
from app.config import settings  # noqa: E402
from app.kb import load_chunks  # noqa: E402


def main():
    chunks = load_chunks(settings.kb_dir)
    if not chunks:
        sys.exit(f"В {settings.kb_dir} нет документов")
    vectors = {}
    for i in range(0, len(chunks), 16):
        batch = chunks[i:i + 16]
        vecs = llm.embed([f"{c.title}\n{c.text}"[:2000] for c in batch])
        if not vecs:
            sys.exit("Сервис эмбеддингов недоступен: проверьте EMBED_API_URL / EMBED_API_KEY")
        vectors.update({c.id: v for c, v in zip(batch, vecs)})
        print(f"{min(i + 16, len(chunks))}/{len(chunks)}", flush=True)
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_path.write_text(json.dumps({"model": settings.embed_model, "vectors": vectors}), encoding="utf-8")
    print("индекс сохранён:", settings.index_path)


if __name__ == "__main__":
    main()

"""База знаний Energytool: загрузка, разбиение на фрагменты, поиск (BM25 + необязательные эмбеддинги).

Поддерживаемые форматы файлов в KB_DIR: .md, .txt, .json, .jsonl, .csv, .pdf (нужен pdfplumber).
Фактов Energytool в коде нет: всё, что бот говорит о продуктах, берётся только из этих файлов.
"""
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import llm
from .config import settings

STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня
еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда
кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда
можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем
хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это нужно хочу хотим""".split())

_SUFFIXES = sorted("""ами ями ого его ому ему ыми ими иях ах ях ов ев ей ой ий ый ая яя ое ее ые ие ую юю ом ем ам ям
ою ею ых их у ю а я ы и е о ь""".split(), key=len, reverse=True)


def stem(word: str) -> str:
    """Очень лёгкий стеммер для русского: отрезает окончание, укорачивает до 6 букв."""
    if not re.search(r"[а-я]", word):
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    return word[:6]


def tokenize(text: str):
    text = (text or "").lower().replace("ё", "е")
    return [stem(w) for w in re.findall(r"[a-zа-я0-9]+", text) if w not in STOP and (len(w) > 1 or w.isdigit())]


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source: str
    tokens: Counter = field(default_factory=Counter, repr=False)
    length: int = 0


def _split_long(text: str, limit: int = 900):
    """Режем длинный текст по абзацам/предложениям, чтобы фрагмент оставался компактным."""
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > limit:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(cur) + len(sent) > limit and cur:
                    parts.append(cur.strip())
                    cur = ""
                cur += sent + " "
            continue
        if len(cur) + len(para) > limit and cur:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _read_markdown(path: Path):
    """Разделы по заголовкам: заголовок становится title фрагмента."""
    out, title, buf = [], path.stem, []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                out.append((title, "\n".join(buf)))
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((title, "\n".join(buf)))
    return out


def _from_obj(obj, default_title):
    if isinstance(obj, dict):
        text = obj.get("text") or obj.get("content") or obj.get("body") or obj.get("description") or ""
        title = obj.get("title") or obj.get("name") or default_title
        extra = [f"{k}: {v}" for k, v in obj.items() if k not in ("text", "content", "body", "title", "name", "description", "id")
                 and isinstance(v, (str, int, float)) and not text]
        return str(title), str(text) or "\n".join(extra)
    return default_title, str(obj)


def _read_file(path: Path):
    ext = path.suffix.lower()
    if ext in (".md", ".markdown"):
        return _read_markdown(path)
    if ext == ".txt":
        return [(path.stem, path.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("documents") or data.get("items") or data.get("data") or [data]
        return [_from_obj(o, path.stem) for o in data]
    if ext == ".jsonl":
        return [_from_obj(json.loads(l), path.stem) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ext == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return [(path.stem, "\n".join(f"{k}: {v}" for k, v in row.items() if v)) for row in csv.DictReader(f)]
    if ext == ".pdf":
        try:
            import pdfplumber  # необязательная зависимость
        except ImportError:
            return []
        with pdfplumber.open(path) as pdf:
            return [(path.stem, "\n\n".join(p.extract_text() or "" for p in pdf.pages))]
    return []


def load_chunks(kb_dir: Path):
    chunks = []
    if not kb_dir.is_dir():
        return chunks
    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file() or path.name.lower().startswith("readme"):
            continue
        try:
            sections = _read_file(path)
        except (OSError, ValueError):
            continue
        n = 0
        for title, text in sections:
            for piece in _split_long(text):
                n += 1
                cid = f"{path.stem}#{n}"
                chunks.append(Chunk(cid, title.strip(), piece, str(path.relative_to(kb_dir))))
    return chunks


def _cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


class KnowledgeBase:
    def __init__(self, chunks, vectors=None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.vectors = vectors or {}  # chunk_id -> вектор (заранее посчитанный индекс)
        df = Counter()
        for c in chunks:
            c.tokens = Counter(tokenize(c.title) * 2 + tokenize(c.text))  # слова заголовка весят вдвое
            c.length = sum(c.tokens.values())
            df.update(c.tokens.keys())
        n = max(len(chunks), 1)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.avg_len = (sum(c.length for c in chunks) / n) or 1.0
        self._embed_cache = lru_cache(maxsize=256)(self._embed_query)

    @classmethod
    def load(cls):
        chunks = load_chunks(settings.kb_dir)
        vectors = {}
        try:
            data = json.loads(settings.index_path.read_text(encoding="utf-8"))
            vectors = {k: v for k, v in data.get("vectors", {}).items() if k in {c.id for c in chunks}}
        except (OSError, ValueError, AttributeError):
            pass
        return cls(chunks, vectors)

    def __len__(self):
        return len(self.chunks)

    # --- поиск ---
    def _bm25(self, query_tokens):
        k1, b = 1.4, 0.75
        scores, mass = {}, sum(self.idf.get(t, 0) for t in set(query_tokens)) or 1.0
        for c in self.chunks:
            s = matched = 0.0
            for t in set(query_tokens):
                f = c.tokens.get(t, 0)
                if f:
                    idf = self.idf[t]
                    s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * c.length / self.avg_len))
                    matched += idf
            if s > 0:
                scores[c.id] = (s, matched / mass)  # (оценка, доля «веса» слов запроса, найденных во фрагменте)
        return scores

    def _embed_query(self, text):
        vecs = llm.embed([text])
        return tuple(vecs[0]) if vecs else None

    def search(self, query: str, top_k=None):
        """Возвращает (список найденных фрагментов, уверенность 0..1, режим поиска)."""
        top_k = top_k or settings.top_k
        qt = tokenize(query)
        bm = self._bm25(qt) if qt else {}
        dense = {}
        if self.vectors:
            qv = self._embed_cache(query[:500])
            if qv:
                dense = {cid: _cosine(qv, v) for cid, v in self.vectors.items()}
        if not bm and not dense:
            return [], 0.0, "none"

        rank = lambda d: {cid: i for i, cid in enumerate(sorted(d, key=d.get, reverse=True))}
        rb = rank({k: v[0] for k, v in bm.items()})
        rd = rank(dense) if dense else {}
        fused = {}
        for cid in set(rb) | set(rd):
            fused[cid] = (1 / (60 + rb[cid]) if cid in rb else 0) + (1 / (60 + rd[cid]) if cid in rd else 0)
        order = sorted(fused, key=fused.get, reverse=True)[:top_k]

        hits = []
        for cid in order:
            c = self.by_id[cid]
            cov = bm.get(cid, (0, 0))[1]
            hits.append({"id": cid, "title": c.title, "source": c.source, "text": c.text,
                         "bm25": round(bm.get(cid, (0, 0))[0], 2), "coverage": round(cov, 2),
                         "dense": round(dense[cid], 3) if cid in dense else None})
        best = hits[0]
        # уверенность: доля значимых слов запроса, найденных в лучшем фрагменте; эмбеддинги добавляют свой сигнал
        confidence = best["coverage"]
        if best["dense"] is not None:
            confidence = max(confidence, min(1.0, max(0.0, (best["dense"] - 0.35) / 0.35)))
        return hits, round(confidence, 2), "hybrid" if dense else "bm25"

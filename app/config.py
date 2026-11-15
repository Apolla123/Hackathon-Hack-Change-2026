"""Настройки читаются только из переменных окружения (см. .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


class Settings:
    # --- языковая модель (OpenAI-совместимый API организаторов) ---
    llm_api_url = property(lambda s: _env("LLM_API_URL"))
    llm_api_key = property(lambda s: _env("LLM_API_KEY"))
    llm_model = property(lambda s: _env("LLM_MODEL", "Qwen3.8-27B"))
    llm_timeout = property(lambda s: _int("LLM_TIMEOUT", 45))
    llm_retries = property(lambda s: _int("LLM_RETRIES", 1))
    llm_max_tokens = property(lambda s: _int("LLM_MAX_TOKENS", 700))
    # 1 = просить сервер не включать «размышления» модели (быстрее). Не все серверы это понимают.
    llm_disable_thinking = property(lambda s: _env("LLM_DISABLE_THINKING", "0") == "1")

    # --- эмбеддинги (необязательно; без них работает BM25) ---
    embed_api_url = property(lambda s: _env("EMBED_API_URL") or _env("LLM_API_URL"))
    embed_api_key = property(lambda s: _env("EMBED_API_KEY") or _env("LLM_API_KEY"))
    embed_model = property(lambda s: _env("EMBED_MODEL", "BAAI/bge-m3"))
    embed_timeout = property(lambda s: _int("EMBED_TIMEOUT", 4))

    # --- база знаний и поиск ---
    kb_dir = property(lambda s: Path(_env("KB_DIR", str(ROOT / "data" / "kb"))))
    index_path = property(lambda s: Path(_env("INDEX_PATH", str(ROOT / "data" / "index" / "embeddings.json"))))
    top_k = property(lambda s: _int("TOP_K", 4))
    min_confidence = property(lambda s: _float("MIN_CONFIDENCE", 0.30))

    # --- хранилище и сервер ---
    db_path = property(lambda s: _env("DB_PATH", str(ROOT / "data" / "energytool.sqlite3")))
    port = property(lambda s: _int("PORT", 8000))
    host = property(lambda s: _env("HOST", "0.0.0.0"))
    cors_origin = property(lambda s: _env("CORS_ORIGIN", ""))  # домен сайта Energytool, если виджет встроен на другом домене
    lead_webhook_url = property(lambda s: _env("LEAD_WEBHOOK_URL"))  # необязательно: куда отправлять заявки
    max_message_chars = property(lambda s: _int("MAX_MESSAGE_CHARS", 2000))
    history_turns = property(lambda s: _int("HISTORY_TURNS", 10))


settings = Settings()

"""Клиент языковой модели и эмбеддингов (OpenAI-совместимый API). Только requests."""
import re
import time

import requests

from .config import settings


class LLMError(Exception):
    pass


def _endpoint(base: str, tail: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith(tail) else base + "/" + tail


def strip_thinking(text: str) -> str:
    # Qwen3 может отдавать рассуждения в <think>...</think>: клиенту они не нужны.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.split("</think>")[-1].strip()


def chat(messages, temperature=0.2, max_tokens=None, json_mode=False):
    """Возвращает (текст, секунды). Повторяет запрос при временных ошибках."""
    if not settings.llm_api_url or not settings.llm_api_key:
        raise LLMError("Не заданы LLM_API_URL / LLM_API_KEY")
    body = {"model": settings.llm_model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if settings.llm_disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    url = _endpoint(settings.llm_api_url, "chat/completions")
    last = "неизвестная ошибка"
    started = time.time()
    for attempt in range(settings.llm_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=settings.llm_timeout)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                content = strip_thinking(content)
                if content:
                    return content, time.time() - started
                last = "пустой ответ модели"
            elif r.status_code in (401, 403):
                raise LLMError("ключ API не подошёл")
            elif r.status_code == 400 and json_mode:
                body.pop("response_format", None)  # сервер не знает json_mode: повторим без него
                last = "HTTP 400"
                continue
            elif r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
            else:
                raise LLMError(f"HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            last = type(e).__name__
        except (KeyError, IndexError, ValueError):
            last = "неожиданный формат ответа"
        if attempt < settings.llm_retries:
            time.sleep(0.5 * (attempt + 1))
    raise LLMError(f"модель недоступна: {last}")


def embed(texts):
    """Векторы для списка текстов или None, если эмбеддинги недоступны (тогда работает BM25)."""
    if not settings.embed_api_url or not settings.embed_api_key or not texts:
        return None
    headers = {"Authorization": f"Bearer {settings.embed_api_key}", "Content-Type": "application/json"}
    try:
        r = requests.post(_endpoint(settings.embed_api_url, "embeddings"), headers=headers,
                          json={"model": settings.embed_model, "input": texts}, timeout=settings.embed_timeout)
        if r.status_code != 200:
            return None
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]
    except (requests.exceptions.RequestException, KeyError, ValueError, TypeError):
        return None

"""solution.py — единая точка входа для автоматической проверки (этап 1).

ВАЖНО: точный формат входа/выхода задан в «Техническом регламенте» организаторов, которого не было среди материалов
при разработке. Здесь реализован гибкий вариант; если регламент требует другое, поправьте только функции
`load_cases` и `save_results` ниже, остальное менять не нужно.

Запуск:  python solution.py --input questions.json --output answers.json
Вход  (JSON-массив или JSONL): элементы вида
    "строка-вопрос"
    {"id": 1, "question": "..."}                         (также ключи query / text / message)
    {"id": 2, "messages": [{"role": "user", "content": "..."}, ...]}   (также history / dialog)
Выход: JSON-массив [{"id": ..., "answer": "..."}], порядок сохраняется.
Каждый тест-кейс обрабатывается независимо: состояние между кейсами не хранится, база данных не используется.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.consultant import Consultant
from app.kb import KnowledgeBase

_consultant = None


def _get_consultant():
    global _consultant
    if _consultant is None:
        _consultant = Consultant(KnowledgeBase.load(), store=None)  # store не нужен: respond() ничего не пишет в БД
    return _consultant


def answer(case) -> str:
    """Ответ бота на одиночный вопрос или на последнюю реплику диалога (с учётом переданной истории)."""
    if isinstance(case, str):
        case = {"question": case}
    msgs = case.get("messages") or case.get("history") or case.get("dialog")
    if msgs:
        msgs = [{"role": "assistant" if str(m.get("role")) in ("assistant", "bot") else "user",
                 "content": str(m.get("content") or m.get("text") or "")} for m in msgs]
        if msgs[-1]["role"] != "user":
            raise ValueError("последняя реплика диалога должна быть от пользователя")
        history, question = msgs[:-1], msgs[-1]["content"]
    else:
        history = []
        question = next((str(case[k]) for k in ("question", "query", "text", "message") if case.get(k)), "")
    return _get_consultant().respond(history, question)["reply"]


def load_cases(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("questions") or data.get("cases") or data.get("data") or [data]
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data


def save_results(path: Path, results):
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_answer(case):
    try:
        return answer(case)
    except Exception as e:  # один сбойный кейс не должен ронять весь прогон
        print(f"ошибка в кейсе: {type(e).__name__}: {e}", file=sys.stderr)
        return "Не удалось обработать вопрос. Предлагаю связаться со специалистом Energytool."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="questions.json")
    p.add_argument("--output", default="answers.json")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    cases = load_cases(Path(args.input))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        answers = list(pool.map(safe_answer, cases))
    results = [{"id": (c.get("id", i) if isinstance(c, dict) else i), "answer": a} for i, (c, a) in enumerate(zip(cases, answers))]
    save_results(Path(args.output), results)
    print(f"готово: {len(results)} ответов -> {args.output}")


if __name__ == "__main__":
    main()

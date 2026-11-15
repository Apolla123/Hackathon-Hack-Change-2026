"""Запуск: python -m unittest discover -s tests -v   (без сети и ключей: модель подменяется заглушкой)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["KB_DIR"] = str(ROOT / "tests" / "fixtures")
os.environ["INDEX_PATH"] = str(Path(tempfile.gettempdir()) / "no_such_index.json")

from app import consultant as cons  # noqa: E402
from app import llm  # noqa: E402
from app.kb import KnowledgeBase, load_chunks, tokenize  # noqa: E402
from app.config import settings  # noqa: E402
from app.store import Store  # noqa: E402


def fake_chat(*replies):
    """Заглушка llm.chat: отдаёт заранее заданные JSON-ответы по очереди."""
    queue = list(replies)
    def _chat(messages, **kw):
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return (item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)), 0.01
    return _chat


class KBTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(load_chunks(settings.kb_dir))

    def test_loaded_by_headings(self):
        self.assertEqual(len(self.kb), 3)

    def test_stemming_finds_word_forms(self):
        hits, conf, mode = self.kb.search("как отключить неплательщика в садовом товариществе")
        self.assertEqual(hits[0]["title"], "Демо-услуга Гамма")
        self.assertEqual(mode, "bm25")

    def test_unrelated_query_has_low_confidence(self):
        hits, conf, _ = self.kb.search("рецепт борща со сметаной")
        self.assertLess(conf, settings.min_confidence)

    def test_empty_kb(self):
        self.assertEqual(KnowledgeBase([]).search("что угодно"), ([], 0.0, "none"))

    def test_tokenize_drops_stopwords(self):
        self.assertNotIn("и", tokenize("счета и тарифы"))


class PhoneTests(unittest.TestCase):
    def test_formats(self):
        for s in ("+7 (916) 123-45-67", "8 916 123 45 67", "9161234567", "89161234567"):
            self.assertEqual(cons.normalize_phone(s), "+79161234567", s)

    def test_rejects_garbage(self):
        self.assertIsNone(cons.normalize_phone("мне 12345 лет"))
        self.assertIsNone(cons.normalize_phone("+7 100 000 00 00"))


class DialogTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.kb = KnowledgeBase(load_chunks(settings.kb_dir))
        self.bot = Consultant = cons.Consultant(self.kb, self.store, summary_async=False)
        self.sid = self.store.new_session()

    def say(self, text, *replies):
        with mock.patch.object(llm, "chat", fake_chat(*replies)):
            return self.bot.handle(self.sid, text)

    def test_answer_with_debug_data(self):
        out = self.say("Что умеет продукт Бета, нужно выбрать тариф для магазина?",
                       {"intent": "answer", "reply": "Бета показывает пиковые часы.", "used": ["demo_kb#2"], "need": "выбрать тариф", "products": ["Бета"]})
        self.assertEqual(out["intent"], "answer")
        d = out["debug"]
        self.assertTrue(d["chunks"] and "total_ms" in d and "retrieval_ms" in d)
        self.assertEqual(d["used_chunks"], ["demo_kb#2"])

    def test_invented_number_triggers_rewrite(self):
        out = self.say("Сколько стоит продукт Альфа?",
                       {"intent": "answer", "reply": "Стоит 99999 рублей.", "used": []},
                       {"intent": "answer", "reply": "В базе знаний указано 1500 условных единиц.", "used": ["demo_kb#1"]})
        self.assertIn("1500", out["reply"])
        self.assertTrue(any("переписан" in w for w in out["debug"]["warnings"]))

    def test_llm_down_uses_kb_fallback(self):
        out = self.say("Расскажи про отключение неплательщиков в садовом товариществе", llm.LLMError("нет сети"))
        self.assertIn("Гамма", out["reply"])
        self.assertIn("llm_error", out["debug"])

    def test_llm_down_and_no_kb_match_offers_specialist(self):
        out = self.say("рецепт борща", llm.LLMError("нет сети"))
        self.assertEqual(out["intent"], "no_answer")
        self.assertIn("специалист", out["reply"])

    def test_non_json_reply_is_not_lost(self):
        out = self.say("Что такое Бета?", "Просто текст без JSON")
        self.assertEqual(out["reply"], "Просто текст без JSON")

    def test_full_lead_flow_and_history(self):
        self.say("Хочу продукт Альфа", {"intent": "offer_lead", "reply": "Хотите, передам менеджеру?", "need": "учёт в здании", "products": ["Альфа"]})
        self.assertEqual(self.store.get_session(self.sid)["lead_stage"], "offered")
        o = self.say("Да, меня зовут Иван Петров", {"intent": "collect_lead", "reply": "Телефон?", "lead_consent": True, "lead": {"name": "Иван Петров"}})
        self.assertEqual(o["lead_stage"], "collecting")
        o = self.say("+7 916 123-45-67", {"intent": "collect_lead", "reply": "Компания?", "lead": {"name": "Иван Петров"}})
        self.assertEqual(o["lead_stage"], "collecting")
        o = self.say("ООО Ромашка", {"intent": "collect_lead", "reply": "ok", "lead": {"company": "ООО Ромашка"}},
                     "Задача клиента: учёт в здании")
        self.assertEqual(o["lead_stage"], "done")
        self.assertIn("L-", o["reply"])
        lead = self.store.list_leads()[0]
        self.assertEqual((lead["name"], lead["phone"], lead["company"]), ("Иван Петров", "+79161234567", "ООО Ромашка"))
        self.assertIn("учёт в здании", lead["summary"])
        self.assertEqual(self.store.list_sessions()[0]["lead_id"], lead["id"])

    def test_hallucinated_name_is_rejected(self):
        self.say("Нужен счётчик", {"intent": "collect_lead", "reply": "Как вас зовут?", "lead_consent": True, "lead": {"name": "Сидор Сидоров", "company": "Газпром"}})
        self.assertEqual(self.store.get_session(self.sid)["lead"], {})

    def test_sessions_are_isolated_and_all_saved(self):
        self.say("Первый вопрос про Альфа", {"intent": "answer", "reply": "ответ 1"})
        sid2 = self.store.new_session()
        with mock.patch.object(llm, "chat") as chat:
            chat.return_value = (json.dumps({"intent": "answer", "reply": "ответ 2"}), 0.01)
            self.bot.handle(sid2, "Второй вопрос про Бета")
            sent = chat.call_args[0][0]
        self.assertNotIn("Первый вопрос", json.dumps(sent, ensure_ascii=False))
        self.assertEqual(len(self.store.list_sessions()), 2)  # сессии без заявок тоже сохранены

    def test_history_goes_to_model(self):
        self.say("Первый вопрос про Альфа", {"intent": "answer", "reply": "ответ 1"})
        with mock.patch.object(llm, "chat") as chat:
            chat.return_value = (json.dumps({"intent": "answer", "reply": "ответ 2"}), 0.01)
            self.bot.handle(self.sid, "а сколько стоит?")
            sent = json.dumps(chat.call_args[0][0], ensure_ascii=False)
        self.assertIn("Первый вопрос", sent)
        self.assertIn("ответ 1", sent)


class SolutionTests(unittest.TestCase):
    def test_single_question_and_dialog(self):
        import solution
        solution._consultant = None
        with mock.patch.object(llm, "chat", fake_chat({"intent": "answer", "reply": "ок"})):
            self.assertEqual(solution.answer("Что такое Бета?"), "ок")
            self.assertEqual(solution.answer({"messages": [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": "Здравствуйте"},
                                                           {"role": "user", "content": "Про Альфа?"}]}), "ок")
            with self.assertRaises(ValueError):
                solution.answer({"messages": [{"role": "assistant", "content": "x"}]})


class ServerTests(unittest.TestCase):
    def test_http_flow(self):
        import threading, urllib.request
        from http.server import ThreadingHTTPServer
        from app.server import App, make_handler
        app = App(KnowledgeBase(load_chunks(settings.kb_dir)), Store(":memory:"))
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        def call(path, body=None):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        try:
            with mock.patch.object(llm, "chat", fake_chat({"intent": "answer", "reply": "привет"})):
                code, out = call("/api/chat", {"message": "Что такое Бета?"})
            self.assertEqual(code, 200)
            sid = out["session_id"]
            code, s = call("/api/sessions/" + sid)
            self.assertEqual([m["role"] for m in s["messages"]], ["user", "assistant"])
            self.assertEqual(call("/api/chat", {"message": "  "})[0], 400)
            self.assertEqual(call("/api/chat", {"message": "x" * 5000})[0], 400)
            self.assertEqual(call("/api/session/new", {})[0], 200)
            with urllib.request.urlopen(base + "/") as r:
                self.assertIn("Energytool", r.read().decode())
            for bad in ("/static/../app/config.py", "/..%2fapp/config.py"):
                try:
                    urllib.request.urlopen(base + bad)
                    self.fail("path traversal")
                except urllib.error.HTTPError as e:
                    self.assertEqual(e.code, 404)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()

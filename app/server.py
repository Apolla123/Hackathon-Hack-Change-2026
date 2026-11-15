"""HTTP-сервер на стандартной библиотеке: чат-API, история сессий, заявки и статика.

Адреса:
  POST /api/chat          {session_id?, message}  -> ответ бота (+ debug)
  POST /api/session/new                           -> новая сессия («Новый диалог»: контекст сбрасывается)
  GET  /api/sessions                              -> история всех сессий (в т.ч. без заявки)
  GET  /api/sessions/<id>                         -> сообщения сессии с отладочными данными
  GET  /api/leads, /api/leads/<id>                -> сформированные заявки с резюме
  GET  /health
"""
import json
import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .config import ROOT, settings
from .consultant import Consultant
from .kb import KnowledgeBase
from .store import Store

STATIC = ROOT / "static"
MAX_BODY = 20_000
_session_locks = {}
_locks_guard = threading.Lock()


def _lock_for(sid):
    with _locks_guard:
        return _session_locks.setdefault(sid, threading.Lock())


class App:
    def __init__(self, kb=None, store=None):
        self.kb = kb or KnowledgeBase.load()
        self.store = store or Store()
        self.consultant = Consultant(self.kb, self.store)

    def chat(self, session_id, message):
        message = (message or "").strip()
        if not message:
            return 400, {"error": "Пустое сообщение"}
        if len(message) > settings.max_message_chars:
            return 400, {"error": f"Сообщение длиннее {settings.max_message_chars} символов"}
        if not session_id or not self.store.get_session(session_id):
            session_id = self.store.new_session()
        with _lock_for(session_id):  # сообщения одной сессии обрабатываются по очереди
            out = self.consultant.handle(session_id, message)
        out["session_id"] = session_id
        return 200, out


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EnergytoolConsultant"

        def log_message(self, fmt, *args):  # не пишем в лог тексты запросов (там могут быть контакты)
            sys.stderr.write("%s %s\n" % (self.command, urlparse(self.path).path))

        def _send(self, code, payload, ctype="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if settings.cors_origin:
                self.send_header("Access-Control-Allow-Origin", settings.cors_origin)
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send(204, b"")

        def _json_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return None
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                return data if isinstance(data, dict) else None
            except ValueError:
                return None

        def do_POST(self):
            path = urlparse(self.path).path
            data = self._json_body()
            if data is None:
                return self._send(400, {"error": "Некорректный запрос"})
            if path == "/api/chat":
                code, out = app.chat(data.get("session_id"), data.get("message"))
                return self._send(code, out)
            if path == "/api/session/new":
                return self._send(200, {"session_id": app.store.new_session()})
            self._send(404, {"error": "not found"})

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                return self._send(200, {"ok": True, "kb_chunks": len(app.kb)})
            if path == "/api/sessions":
                return self._send(200, app.store.list_sessions())
            m = re.fullmatch(r"/api/sessions/([0-9a-f]{12})", path)
            if m:
                s = app.store.get_session(m.group(1))
                if not s:
                    return self._send(404, {"error": "сессия не найдена"})
                s["messages"] = app.store.messages(m.group(1), with_debug=True)
                return self._send(200, s)
            if path == "/api/leads":
                return self._send(200, app.store.list_leads())
            m = re.fullmatch(r"/api/leads/(L-[0-9A-F]{8})", path)
            if m:
                lead = app.store.get_lead(m.group(1))
                return self._send(200 if lead else 404, lead or {"error": "заявка не найдена"})
            self._static(path)

        def _static(self, path):
            rel = "index.html" if path in ("/", "") else path.removeprefix("/static/").lstrip("/")
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():  # защита от ../
                return self._send(404, {"error": "not found"})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text/") or "javascript" in ctype else ""))

    return Handler


def main():
    app = App()
    srv = ThreadingHTTPServer((settings.host, settings.port), make_handler(app))
    print(f"Energytool consultant: http://{settings.host}:{settings.port}  (фрагментов в базе знаний: {len(app.kb)})", flush=True)
    if not len(app.kb):
        print("ВНИМАНИЕ: база знаний пуста. Положите файлы в data/kb/ (см. data/kb/README.md).", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Хранилище диалогов и заявок (SQLite из стандартной библиотеки).

Сохраняются ВСЕ сессии, в том числе без заявки. Контекст между сессиями не смешивается:
история для ответа берётся только из текущей session_id.
"""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, created_at REAL, updated_at REAL,
  lead_stage TEXT DEFAULT 'none', lead_json TEXT DEFAULT '{}', need TEXT DEFAULT '', products_json TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts REAL, debug_json TEXT);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS leads (
  id TEXT PRIMARY KEY, session_id TEXT, created_at REAL, name TEXT, phone TEXT, company TEXT,
  summary TEXT, summary_source TEXT DEFAULT 'template', delivered INTEGER DEFAULT 0);
"""


class Store:
    def __init__(self, path=None):
        self.path = str(path or settings.db_path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _run(self, sql, args=(), commit=True):
        with self._lock:
            cur = self._db.execute(sql, args)
            if commit:
                self._db.commit()
            return cur

    # --- сессии ---
    def new_session(self):
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        self._run("INSERT INTO sessions(id, created_at, updated_at) VALUES (?,?,?)", (sid, now, now))
        return sid

    def get_session(self, sid):
        row = self._run("SELECT * FROM sessions WHERE id=?", (sid,), commit=False).fetchone()
        if not row:
            return None
        d = dict(row)
        d["lead"] = json.loads(d.pop("lead_json") or "{}")
        d["products"] = json.loads(d.pop("products_json") or "[]")
        return d

    def update_session(self, sid, lead_stage=None, lead=None, need=None, products=None):
        s = self.get_session(sid)
        if not s:
            return
        self._run("UPDATE sessions SET lead_stage=?, lead_json=?, need=?, products_json=?, updated_at=? WHERE id=?",
                  (lead_stage if lead_stage is not None else s["lead_stage"],
                   json.dumps(lead if lead is not None else s["lead"], ensure_ascii=False),
                   need if need is not None else s["need"],
                   json.dumps(products if products is not None else s["products"], ensure_ascii=False),
                   time.time(), sid))

    def list_sessions(self):
        rows = self._run("""SELECT s.id, s.created_at, s.updated_at, s.lead_stage, s.need,
            (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS messages,
            (SELECT content FROM messages m WHERE m.session_id=s.id AND m.role='user' ORDER BY m.id LIMIT 1) AS first_question,
            (SELECT id FROM leads l WHERE l.session_id=s.id LIMIT 1) AS lead_id
            FROM sessions s ORDER BY s.updated_at DESC""", commit=False).fetchall()
        return [dict(r) for r in rows]

    # --- сообщения ---
    def add_message(self, sid, role, content, debug=None):
        self._run("INSERT INTO messages(session_id, role, content, ts, debug_json) VALUES (?,?,?,?,?)",
                  (sid, role, content, time.time(), json.dumps(debug, ensure_ascii=False) if debug else None))

    def messages(self, sid, limit=None, with_debug=False):
        rows = self._run("SELECT role, content, ts, debug_json FROM messages WHERE session_id=? ORDER BY id",
                         (sid,), commit=False).fetchall()
        out = []
        for r in rows:
            m = {"role": r["role"], "content": r["content"], "ts": r["ts"]}
            if with_debug and r["debug_json"]:
                m["debug"] = json.loads(r["debug_json"])
            out.append(m)
        return out[-limit:] if limit else out

    # --- заявки ---
    def create_lead(self, sid, name, phone, company, summary, source="template"):
        lid = "L-" + uuid.uuid4().hex[:8].upper()
        self._run("INSERT INTO leads(id, session_id, created_at, name, phone, company, summary, summary_source) VALUES (?,?,?,?,?,?,?,?)",
                  (lid, sid, time.time(), name, phone, company, summary, source))
        return lid

    def update_lead_summary(self, lid, summary, source):
        self._run("UPDATE leads SET summary=?, summary_source=? WHERE id=?", (summary, source, lid))

    def mark_delivered(self, lid):
        self._run("UPDATE leads SET delivered=1 WHERE id=?", (lid,))

    def get_lead(self, lid):
        r = self._run("SELECT * FROM leads WHERE id=?", (lid,), commit=False).fetchone()
        return dict(r) if r else None

    def list_leads(self):
        return [dict(r) for r in self._run("SELECT * FROM leads ORDER BY created_at DESC", commit=False).fetchall()]

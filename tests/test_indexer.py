import json
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from diary_indexer.core import (Settings, IndexerError, Ollama, checkpoint, connect,
    database_lock, discover, normalize_keywords, resolve_date, run, save_entry)


class MockServer:
    def __init__(self):
        self.responses = []
        self.calls = []
        self.callback = None
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.reply(200, {"models": [{"name": "gemma3:4b"}]})
            def do_POST(self):
                owner.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if owner.callback:
                    owner.callback()
                status, value = owner.responses.pop(0) if owner.responses else (200, {"keywords": [" Работа ", "работа", "Москва"]})
                if status == "timeout":
                    time.sleep(0.1)
                    status = 200
                self.reply(status, {"message": {"content": value if isinstance(value, str) else json.dumps(value)}})
            def reply(self, status, body):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pages = self.root / "pages"
        self.pages.mkdir()
        self.settings = Settings(self.pages, self.root / "db.sqlite3", self.root / "cp.jsonl", 2026)
    def note(self, name="1 января чт.md", text="Работа в Москве"):
        path = self.pages / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
    def rows(self, sql="SELECT path FROM entries"):
        with sqlite3.connect(self.settings.database) as db:
            return db.execute(sql).fetchall()

    def test_dates(self):
        valid = {"2026-зима/1 декабря пн.md": "2025-12-01", "2026-зима/1 января чт.md": "2026-01-01", "2024-зима/29 февраля чт.md": "2024-02-29", "архив/2026-лето/1 июня пн.md": "2026-06-01", "2026-весна/1 марта вс.md": "2026-03-01", "2026-осень/1 сентября вт.md": "2026-09-01"}
        for path, expected in valid.items():
            with self.subTest(path=path):
                self.assertEqual(str(resolve_date(Path(path), 2026)), expected)
        self.assertEqual(str(resolve_date(Path("1 января ср.md"), 2025)), "2025-01-01")
        invalid = ["2025-зима/29 февраля сб.md", "1 января пн.md", "1 january чт.md", "misc/1 января чт.md", "2026-зима/sub/1 января чт.md", "2026-зима/2026-зима/1 января чт.md", "2026-лето/1 января чт.md", "32 января чт.md", "hello.md"]
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolve_date(Path(path), 2026)

    def test_order_and_all_errors_without_writes(self):
        for path in ["2026-зима/1 декабря пн.md", "2026-зима/1 января чт.md", "1 января чт.md"]:
            self.note(path)
        self.assertEqual([e.relative for e in discover(self.settings)], ["1 января чт.md", "2026-зима/1 января чт.md", "2026-зима/1 декабря пн.md"])
        self.note("bad.md")
        self.note("wrong.md")
        with self.assertRaises(IndexerError) as error:
            run(self.settings)
        self.assertIn("bad.md", str(error.exception))
        self.assertIn("wrong.md", str(error.exception))
        self.assertFalse(self.settings.database.exists())

    def test_size_context_and_utf8_rejection(self):
        path = self.note(text="я" * 20)
        for settings in [replace(self.settings, max_note_bytes=10), replace(self.settings, context=500)]:
            with self.assertRaises(IndexerError):
                run(settings)
            self.assertFalse(settings.database.exists())
        path.write_bytes(b"\xff")
        with self.assertRaises(IndexerError):
            discover(self.settings)

    def test_configuration_paths(self):
        env = self.root / ".env"
        env.write_text("PAGES_DIR=pages\nCURRENT_YEAR=2024\nMAX_TAGS=5\n")
        settings = Settings.load(env)
        self.assertEqual(settings.pages, self.pages)
        self.assertEqual(settings.year, 2024)
        self.assertEqual(settings.max_tags, 5)

    def test_normalization(self):
        self.assertEqual(normalize_keywords({"keywords": [" МОСКВА ", "москва", "Пешая\n прогулка", "Ａ"]}, 10), ["москва", "пешая прогулка", "a"])
        for value in [{"keywords": [1]}, {"keywords": [" "]}, {"keywords": ["x"], "extra": 1}, {"keywords": "x"}, {"keywords": ["x" * 121]}, {"keywords": ["a\x00b"]}]:
            with self.assertRaises(ValueError):
                normalize_keywords(value, 10)

    def test_index_skip_edit_settings_prune_and_portability(self):
        path = self.note()
        with MockServer() as server:
            settings = replace(self.settings, base_url=server.url)
            run(settings)
            run(settings)
            self.assertEqual(len(server.calls), 1)
            self.assertEqual(self.rows("SELECT name FROM tags ORDER BY name"), [("москва",), ("работа",)])
            self.assertEqual(server.calls[0]["options"]["temperature"], 0)
            self.assertEqual(server.calls[0]["format"]["type"], "object")
            path.write_text("Плавание", encoding="utf-8")
            server.responses = [(200, {"keywords": ["плавание"]})]
            run(settings)
            self.assertEqual(self.rows("SELECT name FROM tags"), [("плавание",)])
            run(replace(settings, max_tags=9))
            self.assertEqual(len(server.calls), 3)
            query = "SELECT e.path FROM entries e JOIN entry_tags et ON et.entry_id=e.id JOIN tags t ON t.id=et.tag_id WHERE t.name='москва' ORDER BY e.diary_date DESC, e.path"
            self.assertEqual(self.rows(query), [(path.name,)])
            copied = self.root / "portable.sqlite3"
            copied.write_bytes(settings.database.read_bytes())
            with sqlite3.connect(copied) as db:
                self.assertEqual(db.execute(query).fetchall(), [(path.name,)])
                self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "delete")
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
            path.unlink()
            run(settings)
            self.assertEqual(len(self.rows()), 1)
            run(settings, "prune")
            self.assertEqual(self.rows(), [])
            self.assertEqual(settings.checkpoint.read_text(), "")

    def test_malformed_and_server_retry(self):
        with MockServer() as server, patch("diary_indexer.core.time.sleep"):
            client = Ollama(replace(self.settings, base_url=server.url))
            self.addCleanup(client.close)
            server.responses = [(500, {}), (503, {}), (200, "not json"), (200, {"keywords": ["Да"]})]
            self.assertEqual(client.extract("text"), ["да"])
            self.assertEqual(len(server.calls), 4)
            server.responses = [(200, "bad"), (200, {"keywords": [3]})]
            with self.assertRaisesRegex(IndexerError, "Malformed"):
                client.extract("text")
            server.responses = [(400, {})]
            before = len(server.calls)
            with self.assertRaises(IndexerError):
                client.extract("text")
            self.assertEqual(len(server.calls), before + 1)

    def test_timeout_retried_twice(self):
        with MockServer() as server:
            client = Ollama(replace(self.settings, base_url=server.url, timeout=0.02))
            self.addCleanup(client.close)
            server.responses = [("timeout", {})] * 3
            with self.assertRaises(IndexerError):
                client.extract("text")
            self.assertEqual(len(server.calls), 3)

    def test_failure_retains_old_keywords_and_source_change(self):
        path = self.note()
        with MockServer() as server, patch("diary_indexer.core.time.sleep"):
            settings = replace(self.settings, base_url=server.url)
            run(settings)
            old = self.rows("SELECT * FROM entries")
            path.write_text("new")
            server.responses = [(200, "bad"), (200, "bad")]
            with self.assertRaises(IndexerError):
                run(settings)
            self.assertEqual(self.rows("SELECT * FROM entries"), old)
            server.callback = lambda: path.write_text("changed again")
            with self.assertRaisesRegex(IndexerError, "Source changed"):
                run(settings)
            self.assertEqual(self.rows("SELECT * FROM entries"), old)

    def test_checkpoint_crash_recovery(self):
        self.note()
        with MockServer() as server:
            settings = replace(self.settings, base_url=server.url)
            real_checkpoint = checkpoint
            calls = []
            def crash(db, path):
                calls.append(1)
                if len(calls) == 2:
                    raise OSError("simulated interruption after commit")
                real_checkpoint(db, path)
            with patch("diary_indexer.core.checkpoint", side_effect=crash), self.assertRaises(OSError):
                run(settings)
            self.assertEqual(len(self.rows()), 1)
            for corrupt in (False, True):
                if corrupt:
                    settings.checkpoint.write_text("corrupt\x00")
                else:
                    settings.checkpoint.unlink()
                run(settings)
                self.assertEqual(len(server.calls), 1)
                record = json.loads(settings.checkpoint.read_text())
                self.assertEqual(record["path"], "1 января чт.md")

    def test_transaction_rollback_and_lock(self):
        self.note()
        entry = discover(self.settings)[0]
        db = connect(self.settings.database)
        self.addCleanup(db.close)
        save_entry(db, entry, "old", ["старое"])
        db.execute("CREATE TRIGGER reject_tag BEFORE INSERT ON tags WHEN NEW.name='fail' BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            save_entry(db, entry, "new", ["fail"])
        self.assertEqual(db.execute("SELECT fingerprint FROM entries").fetchone(), ("old",))
        self.assertEqual(db.execute("SELECT name FROM tags").fetchall(), [("старое",)])
        with database_lock(self.settings.database):
            with self.assertRaisesRegex(IndexerError, "Another indexer"):
                with database_lock(self.settings.database):
                    pass

    def test_validate_and_prune_validate_before_changes(self):
        self.note()
        run(self.settings, "validate")
        self.assertFalse(self.settings.database.exists())
        self.note("bad.md")
        with self.assertRaises(IndexerError):
            run(self.settings, "prune")
        self.assertFalse(self.settings.database.exists())


if __name__ == "__main__":
    unittest.main()

import json
import io
from contextlib import redirect_stderr
import os

import httpx
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from diary_indexer.core import (Settings, IndexerError, RequestLimitReached, Ollama, Anthropic, checkpoint, connect,
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

    def test_short_month_names(self):
        short_names = "янв фев мар апр мая июня июля авг сент окт нояб дек".split()
        weekdays = "чт вс вс ср пт пн ср сб вт чт вс вт".split()
        for month, (name, weekday) in enumerate(zip(short_names, weekdays), 1):
            with self.subTest(month=name):
                self.assertEqual(
                    str(resolve_date(Path(f"1 {name} {weekday}.md"), 2026)),
                    f"2026-{month:02d}-01",
                )
        self.assertEqual(str(resolve_date(Path("2026-зима/1 дек пн.md"), 2026)), "2025-12-01")
        self.assertEqual(str(resolve_date(Path("2024-зима/29 фев чт.md"), 2026)), "2024-02-29")
        for path in ("1 янв пн.md", "2026-лето/1 янв чт.md", "2025-зима/29 фев сб.md"):
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
        env.write_text("PAGES_DIR=pages\nCURRENT_YEAR=2024\nOUTPUT_TOKENS=1024\n")
        settings = Settings.load(env)
        self.assertEqual(settings.pages, self.pages)
        self.assertEqual(settings.year, 2024)
        self.assertEqual(settings.output_tokens, 1024)

    def anthropic_client(self, responses):
        settings = replace(self.settings, provider="anthropic", api_key="test-secret",
                           base_url="https://api.anthropic.com", model="claude-haiku-4-5-20251001")
        client = Anthropic(settings)
        client.client.close()
        calls = []
        def handler(request):
            calls.append(request)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            status, body = response
            return httpx.Response(status, json=body)
        client.client = httpx.Client(base_url=settings.base_url,
            headers={"x-api-key": settings.api_key, "anthropic-version": "2023-06-01"},
            transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return settings, client, calls

    def anthropic_response(self, value, reason="end_turn"):
        return {"stop_reason": reason, "content": [{"type": "text", "text": json.dumps(value)}]}

    def test_anthropic_configuration_and_fingerprint(self):
        env = self.root / ".env"
        env.write_text("EXTRACTION_PROVIDER=anthropic\nANTHROPIC_API_KEY=file-key\n")
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.load(env)
            self.assertEqual(settings.provider, "anthropic")
            self.assertEqual(settings.base_url, "https://api.anthropic.com")
            self.assertEqual(settings.model, "claude-haiku-4-5-20251001")
            self.assertEqual(settings.api_key, "file-key")
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "override", "ANTHROPIC_MODEL": "other"}):
                overridden = Settings.load(env)
            self.assertEqual(overridden.api_key, "override")
            self.assertEqual(overridden.model, "other")
            env.write_text("EXTRACTION_PROVIDER=unknown\n")
            with self.assertRaisesRegex(IndexerError, "EXTRACTION_PROVIDER"):
                Settings.load(env)
        self.assertNotIn("file-key", repr(settings))
        self.assertEqual(settings.fingerprint, replace(settings, api_key="new-key").fingerprint)
        self.assertNotEqual(settings.fingerprint, replace(settings, provider="ollama").fingerprint)
        self.assertNotEqual(settings.fingerprint, replace(settings, model="other").fingerprint)

    def test_anthropic_diagnostics_show_invalid_keywords_before_failure(self):
        body = self.anthropic_response({"keywords": [""]})
        _, client, calls = self.anthropic_client([(200, body), (200, body)])
        output = io.StringIO()
        with redirect_stderr(output), patch("diary_indexer.core.time.perf_counter",
                side_effect=[0, 2.5, 10, 14]):
            with self.assertRaisesRegex(IndexerError, "nonempty"):
                client.extract("note")
        log = output.getvalue()
        self.assertEqual(len(calls), 2)
        self.assertEqual(log.count("Anthropic raw response:"), 2)
        self.assertIn("completed in 2.50s (HTTP 200)", log)
        self.assertIn("completed in 4.00s (HTTP 200)", log)
        self.assertIn("keywords", log)
        self.assertIn("end_turn", log)
        self.assertNotIn("test-secret", log)

    def test_anthropic_request_and_normalization(self):
        settings = replace(self.settings, provider="anthropic", api_key="test-secret")
        real_client = Anthropic(settings)
        self.addCleanup(real_client.close)
        self.assertEqual(real_client.client.headers["x-api-key"], "test-secret")
        self.assertEqual(real_client.client.headers["anthropic-version"], "2023-06-01")
        settings, client, calls = self.anthropic_client([
            (200, self.anthropic_response({"keywords": [" МОСКВА ", "москва", "Ａ"]}))])
        self.assertEqual(client.extract("Дневник"), ["москва", "a"])
        request = calls[0]
        self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
        body = json.loads(request.content)
        self.assertEqual(body["model"], settings.model)
        self.assertEqual(body["max_tokens"], settings.output_tokens)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["messages"], [settings.messages("Дневник")[1]])
        self.assertEqual(body["system"], settings.messages("Дневник")[0]["content"])
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("minLength", json.dumps(body["output_config"]))
        self.assertNotIn("test-secret", request.content.decode())

    def test_anthropic_retries_and_rejections(self):
        good = self.anthropic_response({"keywords": ["Да"]})
        with patch("diary_indexer.core.time.sleep"):
            _, client, calls = self.anthropic_client([
                (429, {}), (529, {}), (200, self.anthropic_response({"keywords": [1]})), (200, good)])
            self.assertEqual(client.extract("text"), ["да"])
            self.assertEqual(len(calls), 4)
            for status in (400, 401, 403, 404):
                _, client, calls = self.anthropic_client([(status, {})])
                with self.assertRaisesRegex(IndexerError, "Anthropic request failed"):
                    client.extract("text")
                self.assertEqual(len(calls), 1)
            _, client, calls = self.anthropic_client([httpx.ReadTimeout("timeout")] * 3)
            with self.assertRaisesRegex(IndexerError, "Anthropic request failed"):
                client.extract("text")
            self.assertEqual(len(calls), 3)
            bad_bodies = [None, {}, {"stop_reason": "end_turn", "content": []},
                self.anthropic_response({"keywords": []}, "max_tokens"),
                self.anthropic_response({"keywords": ["x" * 121]}),
                self.anthropic_response({"keywords": [""]})]
            for body in bad_bodies:
                with self.subTest(body=body):
                    _, client, calls = self.anthropic_client([(200, body)] * 2)
                    with self.assertRaisesRegex(IndexerError, "Malformed Anthropic"):
                        client.extract("text")
                    self.assertEqual(len(calls), 2)
            _, client, calls = self.anthropic_client([(200, self.anthropic_response({}, "refusal"))])
            with self.assertRaisesRegex(IndexerError, "refused"):
                client.extract("text")
            self.assertEqual(len(calls), 1)

    def test_anthropic_resume_provider_switch_and_failure_preservation(self):
        self.note()
        entry = discover(self.settings)[0]
        db = connect(self.settings.database)
        save_entry(db, entry, self.settings.fingerprint, ["старое"])
        db.close()
        responses = [(200, self.anthropic_response({"keywords": ["Новое"]}))]
        settings, client, calls = self.anthropic_client(responses)
        # Keep the mocked client open across runs; run must still call close.
        with patch("diary_indexer.core.Anthropic", return_value=client), patch.object(client, "close") as close:
            run(settings)
            run(settings)
            self.assertEqual(len(calls), 1)
            self.assertEqual(close.call_count, 2)
            self.assertEqual(self.rows("SELECT name FROM tags"), [("новое",)])
            old = self.rows("SELECT * FROM entries")
            responses.extend([(200, self.anthropic_response({}, "refusal"))])
            with self.assertRaisesRegex(IndexerError, "refused"):
                run(replace(settings, output_tokens=1024))
            self.assertEqual(self.rows("SELECT * FROM entries"), old)
            self.assertEqual(self.rows("SELECT name FROM tags"), [("новое",)])
            record = json.loads(settings.checkpoint.read_text())
            self.assertEqual(record["fingerprint"], settings.fingerprint)

    def test_anthropic_offline_commands_need_no_key(self):
        self.note()
        settings = replace(self.settings, provider="anthropic")
        with patch("diary_indexer.core.httpx.Client") as client:
            run(settings, "validate")
            self.assertFalse(settings.database.exists())
            run(settings, "prune")
            with self.assertRaisesRegex(IndexerError, "ANTHROPIC_API_KEY"):
                run(settings)
            client.assert_not_called()

    def test_request_limit_counts_retries_for_both_providers(self):
        for provider in ("ollama", "anthropic"):
            for status in (200, 503):
                with self.subTest(provider=provider, status=status):
                    settings = replace(self.settings, provider=provider, api_key="test-key")
                    client = (Ollama if provider == "ollama" else Anthropic)(settings)
                    self.addCleanup(client.close)
                    client.max_requests = 1
                    # Invalid output normally causes another extraction request;
                    # a server failure normally causes a transport retry.
                    response = httpx.Response(status, json={}, request=httpx.Request("POST", "https://example.test"))
                    with patch.object(client.client, "request", return_value=response) as request, patch(
                            "diary_indexer.core.time.sleep"):
                        with self.assertRaises(RequestLimitReached):
                            client.extract("note")
                    self.assertEqual(request.call_count, 1)
                    self.assertEqual(client.requests_used, 1)

    def test_request_limit_preserves_progress_and_resumes(self):
        self.note("1 января чт.md")
        self.note("2 января пт.md")
        with MockServer() as server:
            settings = replace(self.settings, base_url=server.url)
            run(replace(settings, max_requests=1))
            self.assertEqual(len(server.calls), 1)
            self.assertEqual(self.rows(), [("2 января пт.md",)])
            self.assertEqual(len(settings.checkpoint.read_text().splitlines()), 1)
            run(replace(settings, max_requests=1))
            self.assertEqual(len(server.calls), 2)
            self.assertEqual(len(self.rows()), 2)
            run(replace(settings, max_requests=1))
            self.assertEqual(len(server.calls), 2)

    def test_request_limit_configuration(self):
        env = self.root / ".env"
        with patch.dict(os.environ, {}, clear=True):
            for value in ("0", "-1", "1.5", "abc"):
                env.write_text(f"MAX_REQUESTS={value}\n")
                with self.subTest(value=value), self.assertRaisesRegex(IndexerError, "MAX_REQUESTS"):
                    Settings.load(env)
            for contents in ("", "MAX_REQUESTS=\n", "MAX_REQUESTS=   \n"):
                env.write_text(contents)
                self.assertIsNone(Settings.load(env).max_requests)
            env.write_text("MAX_REQUESTS=2\n")
            settings = Settings.load(env)
            self.assertEqual(settings.max_requests, 2)
            self.assertEqual(settings.fingerprint, replace(settings, max_requests=None).fingerprint)
            with patch.dict(os.environ, {"MAX_REQUESTS": "3"}):
                self.assertEqual(Settings.load(env).max_requests, 3)
            with patch.dict(os.environ, {"MAX_REQUESTS": ""}):
                self.assertIsNone(Settings.load(env).max_requests)
        self.note()
        run(replace(self.settings, max_requests=1), "validate")
        run(replace(self.settings, max_requests=1), "prune")

    def test_legacy_max_tags_is_ignored(self):
        env = self.root / ".env"
        env.write_text("MAX_TAGS=1\n")
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.load(env)
        self.assertEqual(settings.output_tokens, 2048)
        self.assertNotIn("maxItems", settings.schema["properties"]["keywords"])

    def test_keyword_count_is_unrestricted(self):
        keywords = [f"тема {i}" for i in range(100)]
        self.assertEqual(normalize_keywords({"keywords": keywords}), keywords)
        for provider in ("ollama", "anthropic"):
            settings = replace(self.settings, provider=provider, api_key="test-key")
            self.assertNotIn("maxItems", json.dumps(settings.request_payload("note")))
            self.assertNotIn("не должно превышать", settings.messages("note")[0]["content"])
            client = (Ollama if provider == "ollama" else Anthropic)(settings)
            self.addCleanup(client.close)
            body = ({"message": {"content": json.dumps({"keywords": keywords})}}
                    if provider == "ollama" else self.anthropic_response({"keywords": keywords}))
            with patch.object(client, "request", return_value=httpx.Response(200, json=body)):
                self.assertEqual(client.extract("note"), keywords)

    def test_normalization(self):
        self.assertEqual(normalize_keywords({"keywords": [" МОСКВА ", "москва", "Пешая\n прогулка", "Ａ"]}), ["москва", "пешая прогулка", "a"])
        for value in [{"keywords": [1]}, {"keywords": [" "]}, {"keywords": ["x"], "extra": 1}, {"keywords": "x"}, {"keywords": ["x" * 121]}, {"keywords": ["a\x00b"]}]:
            with self.assertRaises(ValueError):
                normalize_keywords(value)

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
            run(replace(settings, output_tokens=1024))
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

    def test_ollama_diagnostics_include_raw_responses_and_elapsed_time(self):
        client = Ollama(self.settings)
        client.client.close()
        replies = [httpx.Response(503, text="temporarily unavailable"),
                   httpx.Response(200, text="not JSON: Привет")]
        client.client = httpx.Client(base_url=self.settings.base_url,
            transport=httpx.MockTransport(lambda request: replies.pop(0)))
        self.addCleanup(client.close)
        output = io.StringIO()
        with redirect_stderr(output), patch("diary_indexer.core.time.sleep"), patch(
                "diary_indexer.core.time.perf_counter", side_effect=[10, 12.5, 20, 24]):
            response = client.request("POST", "/api/chat")
        self.assertEqual(response.text, "not JSON: Привет")
        log = output.getvalue()
        self.assertIn("completed in 2.50s (HTTP 503)", log)
        self.assertIn("temporarily unavailable", log)
        self.assertIn("completed in 4.00s (HTTP 200)", log)
        self.assertIn("Ollama raw response:\nnot JSON: Привет", log)

    def test_ollama_diagnostics_time_transport_failures(self):
        client = Ollama(self.settings)
        self.addCleanup(client.close)
        output = io.StringIO()
        with redirect_stderr(output), patch.object(client.client, "request",
                side_effect=httpx.ReadTimeout("timed out")), patch(
                "diary_indexer.core.time.sleep"), patch(
                "diary_indexer.core.time.perf_counter", side_effect=[0, 5, 10, 15, 20, 25]):
            with self.assertRaises(IndexerError):
                client.request("POST", "/api/chat")
        self.assertEqual(output.getvalue().count("failed after 5.00s: ReadTimeout"), 3)

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

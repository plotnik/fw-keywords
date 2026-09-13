Streamlit troubleshooting
=========================

Preview the next pending note using the indexer's own validation and request
builder. Extraction runs only on a button click; results stay in this browser
session and never update SQLite or the checkpoint.

::

  import json
  import sqlite3
  import time
  from contextlib import closing
  from pathlib import Path

  import streamlit as st

  from diary_indexer.core import (
      Anthropic, IndexerError, Ollama, RequestLimitReached, Settings,
      digest, discover, read_note,
  )


.. function:: pending_entries(settings, entries)

::

  def pending_entries(settings, entries):
      """Inspect resume metadata without creating or modifying a database."""
      if not settings.database.exists():
          return entries
      uri = settings.database.resolve().as_uri() + "?mode=ro"
      with closing(sqlite3.connect(uri, uri=True)) as db:
          version = db.execute("PRAGMA user_version").fetchone()[0]
          if version != 1:
              raise IndexerError(f"Unsupported database schema version {version}")
          previous = {row[0]: row[1:] for row in db.execute(
              "SELECT path, content_hash, fingerprint, diary_date FROM entries")}
      return [entry for entry in entries if previous.get(entry.relative) != (
          entry.content_hash, settings.fingerprint, entry.diary_date)]


.. function:: extract(settings, entry, note)

::

  def extract(settings, entry, note):
      """Keep every HTTP response, including failures and malformed outputs."""
      result = {"responses": [], "keywords": None, "error": None}
      started = time.perf_counter()

      def capture(response):
          response.read()
          result["responses"].append({
              "path": response.request.url.path,
              "status": response.status_code, "body": response.text,
          })

      client = None
      try:
          if digest(read_note(entry.path, settings)) != entry.content_hash:
              raise IndexerError("Source changed since preview; reload the files.")
          client = (Anthropic if settings.provider == "anthropic" else Ollama)(settings)
          client.client.event_hooks["response"].append(capture)
          client.check_model()
          result["keywords"] = client.extract(note)
      except RequestLimitReached:
          result["error"] = f"MAX_REQUESTS={settings.max_requests} reached during retries."
      except (IndexerError, OSError, ValueError) as exc:
          result["error"] = str(exc)
      finally:
          if client is not None:
              client.close()
          result["elapsed"] = time.perf_counter() - started
      return result


.. function:: render_text(text, as_json)

::

  def render_text(text, as_json):
      if as_json:
          try:
              st.json(json.loads(text))
              return
          except ValueError:
              st.warning("This result is not valid JSON; showing the original text.")
      st.code(text, language=None)


.. function:: main()

::

  def main():
      st.set_page_config(page_title="Diary LLM troubleshooting", layout="wide")
      st.title("Diary LLM troubleshooting")
      st.caption("Preview and test extraction. The database and checkpoint are not updated.")
      env = st.text_input("Configuration file", str(Path(__file__).parent / ".env"))
      st.button("Reload files and configuration")
      try:
          settings = Settings.load(Path(env))
          entries = discover(settings)
          pending = pending_entries(settings, entries)
          st.caption(f"{len(entries)} files · {len(pending)} pending · {settings.provider} · {settings.model}")
          st.text(f"Endpoint: {settings.base_url}")
          all_files = st.checkbox("Include already indexed files")
          choices = entries if all_files else pending
          if not choices:
              st.info("No pending files. Include already indexed files to retest one."
                      if entries else "No Markdown diary files found.")
              return
          paths = [entry.relative for entry in choices]
          selected = st.selectbox("File to process (newest first)", paths)
          entry = choices[paths.index(selected)]
          content = read_note(entry.path, settings)
          if digest(content) != entry.content_hash:
              raise IndexerError("Source changed during discovery; reload the files.")
          note = content.decode("utf-8")
      except (IndexerError, OSError, ValueError, sqlite3.Error) as exc:
          st.error(str(exc))
          return

      st.text(f"File: {entry.path}\nDiary date: {entry.diary_date}")
      st.subheader("Populated prompt")
      for message in settings.messages(note):
          st.caption(message["role"])
          st.code(message["content"], language=None)
      with st.expander("Exact request body"):
          st.json(settings.request_payload(note))

      # Bind the stored result to the displayed source and settings. Rerendering
      # the text/JSON toggle must never submit another request or show stale output.
      identity = (settings, entry)
      if st.session_state.get("preview_identity") != identity:
          st.session_state.pop("result", None)
          st.session_state["preview_identity"] = identity
      st.caption("Send uses the configured provider, retries and MAX_REQUESTS limit. "
                 "Anthropic requests incur API usage charges.")
      if st.button("Send request to LLM", type="primary"):
          with st.spinner("Waiting for the model…"):
              st.session_state["result"] = extract(settings, entry, note)

      as_json = st.toggle("Display result as JSON", value=False)
      result = st.session_state.get("result")
      if result is None:
          return
      st.subheader("LLM result")
      st.caption(f"Elapsed: {result['elapsed']:.2f}s")
      if result["error"]:
          st.error(result["error"])
      responses = result["responses"]
      for number, response in enumerate(responses, 1):
          st.caption(f"Response {number} · {response['path']} · HTTP {response['status']}")
          raw = response["body"]
          try:
              body = json.loads(raw)
              if settings.provider == "ollama" and response["path"].endswith("/api/chat"):
                  raw = body["message"]["content"]
              elif settings.provider == "anthropic":
                  raw = "".join(block["text"] for block in body["content"] if block["type"] == "text")
          except (ValueError, KeyError, TypeError):
              pass
          render_text(raw, as_json)
          with st.expander(f"Full HTTP response {number}"):
              render_text(response["body"], as_json)
      if result["keywords"] is not None:
          st.caption("Validated, normalized keywords")
          render_text(json.dumps({"keywords": result["keywords"]}, ensure_ascii=False), as_json)


  if __name__ == "__main__":
      main()

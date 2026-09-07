# Diary keyword indexer

Python 3.11+ CLI that indexes Russian Markdown diary entries sequentially, newest first, using a local Ollama model. SQLite stores keywords and metadata; diary text is sent only to your configured Ollama endpoint and is not stored in the database or checkpoint.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
ollama pull gemma3:4b
```

Install Ollama separately and start its service (`ollama serve` if needed). Model downloading is always manual. Before extraction, the CLI checks `/api/tags` for the configured model. No external AI service is used by default. Keep `OLLAMA_BASE_URL` pointing to your local Ollama server.

Edit `.env` to set `PAGES_DIR`. Relative paths resolve against the selected `.env` file's directory, regardless of the working directory. Process environment variables override `.env`. A missing `.env` uses defaults relative to its intended location.

| Setting | Default | Meaning |
| --- | --- | --- |
| `PAGES_DIR` | `pages` | Source directory |
| `DATABASE_PATH` | `diary.sqlite3` | Portable SQLite output |
| `CHECKPOINT_PATH` | `checkpoint.jsonl` | Readable recovery mirror |
| `CURRENT_YEAR` | local year at startup | Year for root entries; blank uses default |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `gemma3:4b` | Installed model name |
| `REQUEST_TIMEOUT` | `600` | Seconds per HTTP operation |
| `CONTEXT_SIZE` | `16384` | Model context tokens |
| `MAX_NOTE_BYTES` | `12000` | Maximum UTF-8 note size |
| `MAX_TAGS` | `10` | Maximum returned keywords |

The defaults target a 16 GB machine; actual memory use depends on model quantization and other running software. Only one note is submitted at a time.

## Validate and index

```sh
python -m diary_indexer validate
python -m diary_indexer
python -m diary_indexer index --env /absolute/path/to/.env
python -m diary_indexer prune
```

`validate` performs no Ollama calls and writes no indexing files. All commands validate the complete source tree before changing indexing records. Errors list invalid paths. Notes must be valid UTF-8. Non-`.md` files are ignored; symbolic-link notes and directories are rejected.

Names are case-sensitive: `day Russian-month weekday.md`, for example `1 января чт.md`. Months use Russian genitive names (`января` through `декабря`); weekdays are `пн вт ср чт пт сб вс`. Root entries use `CURRENT_YEAR`. Nested entries must be directly inside exactly one `YYYY-зима`, `YYYY-весна`, `YYYY-лето`, or `YYYY-осень` ancestor. Organizational folders *above* that ancestor are allowed; folders below it are rejected.

`2026-зима/1 декабря пн.md` resolves to `2025-12-01`; January and February in that season resolve to 2026. Calendar dates, weekdays and season/month agreement are checked. Entries sort by date descending and then relative POSIX path ascending. Duplicate dates are allowed.

Oversized notes fail validation; nothing is truncated or split. A second conservative budget counts the full serialized prompt in UTF-8 bytes at one byte per token, adds 256 tokens for chat framing, and reserves `max(512, MAX_TAGS * 64)` tokens for output. This intentionally rejects some notes that might fit with a particular tokenizer. Increase configured limits explicitly if needed.

Extraction uses a Russian system prompt and Ollama's [JSON-schema structured outputs](https://docs.ollama.com/capabilities/structured-outputs), with temperature zero. Diary content is supplied as data in a separate user message. Returned keywords are validated locally, NFKC-normalized, case-folded, whitespace-normalized and deduplicated. Synonyms and `е`/`ё` are not merged. Dictionary forms and Russian topics, activities, people and places are requested but remain model-dependent. Empty keyword lists are valid.

## Recovery and updates

SQLite is authoritative. An entry is skipped when its content SHA-256, extraction fingerprint and resolved date match committed metadata. The fingerprint includes the model name, prompt/version, schema, normalization version, context, output budget and extraction limits. Replacing model weights under the same model name is not detected; use a distinct model name to force reindexing. Changing `CURRENT_YEAR` updates affected dates.

Each successful extraction replaces an entry and its tag relationships in one SQLite transaction. Failure leaves previous successful keywords intact. After every commit, a UTF-8 JSONL checkpoint is written to a temporary file, flushed, and atomically replaced. Each line contains `path`, `content_hash`, `fingerprint` and `indexed_at`. At startup it is rebuilt from SQLite, so missing/corrupt checkpoints and interruption after commit do not repeat successful extraction. Do not edit the checkpoint to control indexing.

Connection failures, timeouts, HTTP 408/429 and server errors get two retries, waiting one and two seconds. Malformed output gets one additional model request; transport retries apply to each request. Exhausted retries stop the run. Correct the cause and rerun the same command. Ctrl-C exits with code 130; other handled failures exit with code 1.

Source hashes are checked again before processing and immediately after extraction. Detected edits or deletions stop processing before stale results are committed. Avoid editing the source tree during a run: filesystem changes cannot be locked atomically together with SQLite, and new files are discovered on the next run.

A persistent `.lock` sidecar prevents concurrent indexers using the same database path. The OS releases the lock when the process exits; do not delete the sidecar while running. Use the same canonical database path and avoid hard-link aliases, network filesystems and sharing one checkpoint between different databases.

Missing source entries are retained by indexing. `prune` validates the source tree, explicitly removes missing paths and unused tags, then refreshes the checkpoint. Check `PAGES_DIR` before pruning: an existing empty directory removes all entries.

## Database transfer and Spring Boot searches

Schema version is in `PRAGMA user_version` (currently 1). Unknown versions are rejected. Tables:

- `entries`: `id`, unique relative `path`, ISO `diary_date`, `content_hash`, extraction `fingerprint`, UTC `indexed_at`.
- `tags`: `id`, unique normalized `name`.
- `entry_tags`: unique `(entry_id, tag_id)` with foreign keys and a `(tag_id, entry_id)` lookup index.

The database uses SQLite's standard DELETE rollback journal and FULL synchronous writes. **Copy the database to the Pi only after the indexer exits.** Do not copy an active database. If a process crashed, rerun successfully first so SQLite can recover any hot journal. The checkpoint and lock file are unnecessary for searching. Paths contain no machine-specific source prefix; resolve them against the Pi's diary root.

Exact normalized-tag search, newest first:

```sql
SELECT e.path, e.diary_date
FROM tags t
JOIN entry_tags et ON et.tag_id = t.id
JOIN entries e ON e.id = et.entry_id
WHERE t.name = ?
ORDER BY e.diary_date DESC, e.path ASC;
```

For Spring Boot, configure a SQLite JDBC driver and use the same parameterized SQL through `JdbcTemplate`:

```java
List<String> paths = jdbcTemplate.query(
    """
    SELECT e.path
    FROM tags t
    JOIN entry_tags et ON et.tag_id = t.id
    JOIN entries e ON e.id = et.entry_id
    WHERE t.name = ?
    ORDER BY e.diary_date DESC, e.path ASC
    """,
    (rs, rowNum) -> rs.getString("path"),
    normalizedTag
);
```

Supply the stored normalized tag (for example `москва`). For user-entered search terms, implement equivalent Unicode NFKC, case folding and whitespace normalization; Java lowercase alone is not fully equivalent to Unicode case folding. The UI may instead offer existing `tags.name` values. SQLite's default `lower()` is not sufficient for Cyrillic normalization. Full Spring Boot integration is outside this project.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use a local mock HTTP server and temporary files, never a real model. They cover dates and ordering, configuration, normalization, size/context rejection, schema requests, retries/timeouts, skips and replacements, fingerprint changes, failure preservation, source edits, transaction rollback, checkpoint recovery, locks, pruning and Cyrillic database portability.

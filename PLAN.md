# Resumable diary keyword indexer

## Summary

Create a standalone Python CLI that runs on the 16 GB computer, reads Russian Markdown diary entries, extracts keywords using local Ollama, and writes a portable SQLite database for future Spring Boot searches.

Process entries sequentially, newest first. Maintain a readable JSON Lines checkpoint file and recover safely after interruptions.

## Discovery and date validation

- Recursively collect `.md` files under the configured `pages` directory at startup.
- Parse filenames as `day Russian-month weekday.md`, using Russian month names and weekday abbreviations `пн–вс`.
- Root-level files use `CURRENT_YEAR`, defaulting to the local calendar year captured at startup.
- Nested files must have exactly one season ancestor named `YYYY-зима`, `YYYY-весна`, `YYYY-лето`, or `YYYY-осень`. Additional organizational folders below that ancestor are not allowed.
- Winter December belongs to the preceding year: `2026-зима/1 декабря пн.md` means `2025-12-01`. January and February belong to 2026.
- Validate real dates, season/month consistency, and weekday agreement for every entry before making any LLM calls or changing indexing records. Report all invalid paths and exit unsuccessfully.
- Sort by resolved date descending, with relative path as a deterministic tie-breaker. Multiple entries may share a date.

## Extraction and configuration

- Use Python 3.11+, `python-dotenv`, an HTTP client, and Python’s built-in SQLite support.
- Provide `python -m diary_indexer` for indexing and `python -m diary_indexer validate` for validation without Ollama calls.
- Read application settings from `.env`; resolve relative paths against that file’s directory. Include a documented `.env.example`.
- Configure `PAGES_DIR`, `DATABASE_PATH`, `CHECKPOINT_PATH`, `CURRENT_YEAR`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, request timeout, context size, maximum note bytes, and maximum tags.
- Default to `http://localhost:11434`, `gemma3:4b`, a 600-second request timeout, 16,384 context tokens, 12,000 note bytes, and 10 tags.
- Send one note per request with a Russian extraction prompt, temperature zero, and a JSON schema for `{"keywords": [...]}`. Validate responses locally. [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- Request concise Russian topics, activities, people, and places, preferring dictionary forms. Normalize Unicode, letter case, and whitespace; deduplicate tags. Do not merge synonyms automatically.
- Treat diary content as data, never as instructions. Store keywords and indexing metadata, not diary text.
- Reject oversized notes before indexing begins; never silently truncate or split them. Also enforce a conservative full-prompt byte budget against the configured context size, reserving space for output.
- Retry transient connection/server failures twice with backoff. Allow one retry for malformed model output, then stop with an actionable error. Preserve completed work.

## SQLite and interruption recovery

- Use normalized tables:
  - `entries`: unique relative POSIX path, ISO diary date, content SHA-256, extraction-settings fingerprint, indexing timestamp.
  - `tags`: unique normalized tag name.
  - `entry_tags`: unique entry/tag relationships, foreign keys, and an index supporting tag-to-entry lookup.
- Include schema versioning and a documented SQL query returning matching paths newest first. Spring Boot integration itself is outside this project.
- Identify entries by relative path, making the database independent of computer-specific directories.
- Skip entries only when both content hash and extraction fingerprint match committed SQLite metadata. Changes to model, prompt version, or extraction settings trigger reindexing.
- Replace an entry’s keywords in one transaction, retaining its previous successful keywords if extraction fails.
- After each database commit, atomically replace the UTF-8 JSONL checkpoint containing successful paths, hashes, fingerprints, and timestamps.
- Reconcile the checkpoint from SQLite at startup. An interruption between database commit and checkpoint update therefore does not repeat a completed extraction.
- Detect source changes during extraction and stop without committing stale results. Prevent concurrent indexers sharing the same database.
- Keep missing entries by default. Provide an explicit `prune` command that validates the source tree, removes missing paths, and refreshes the checkpoint.
- Use SQLite’s standard rollback journal mode and document copying the database to the Pi only after the indexer exits.

## Validation and delivery

- Test winter boundaries, leap years, Russian names, root-year overrides, invalid weekdays, unknown paths, and newest-first ordering.
- Use a mock Ollama server to test valid/invalid output, timeouts, retries, and oversized-note rejection.
- Test unchanged skips, edited-note replacement, settings changes, transaction rollback, missing/corrupt checkpoints, crash recovery, concurrent-run rejection, and explicit pruning.
- Verify tag-search SQL and portability with Cyrillic paths.
- Deliver the CLI, dependency configuration, tests, `.env.example`, and README covering setup, model installation, validation, indexing, recovery, database transfer, and Spring Boot query examples.
- Assume local model availability is checked before indexing; no automatic model downloads or external AI services.

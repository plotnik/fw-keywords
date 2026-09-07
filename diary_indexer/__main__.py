import argparse
import sqlite3
import sys
from pathlib import Path

from .core import IndexerError, Settings, run


def main():
    parser = argparse.ArgumentParser(description="Index Russian diary keywords using local Ollama")
    parser.add_argument("command", nargs="?", choices=("index", "validate", "prune"), default="index")
    parser.add_argument("--env", type=Path, default=Path(".env"), help="settings file (default: ./.env)")
    args = parser.parse_args()
    try:
        run(Settings.load(args.env), args.command)
    except (IndexerError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed SQLite transactions are safe; rerun to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

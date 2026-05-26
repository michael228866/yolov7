"""SQLite-backed event log for the head-counter.

Schema:
  sessions  : one row per program run
  events    : one row per line crossing
Indexes optimise day-level queries.

All timestamps are stored as ISO-8601 strings in *local* time (no TZ suffix),
because the use case is "how many people came in today" measured in the
camera's wall-clock, not UTC.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / 'counter.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    source      TEXT,
    weights     TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    direction   TEXT NOT NULL CHECK(direction IN ('in','out')),
    track_id    INTEGER NOT NULL,
    session_id  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(substr(ts, 1, 10));
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


@contextmanager
def connect(path: Path = DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA foreign_keys = ON;')
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def start_session(source: str, weights: str, notes: str = '',
                  path: Path = DB_PATH) -> str:
    sid = uuid.uuid4().hex[:12]
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO sessions (id, started_at, source, weights, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, now_iso(), source, weights, notes),
        )
    return sid


def end_session(session_id: str, path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (now_iso(), session_id),
        )


def log_events(rows, path: Path = DB_PATH) -> None:
    """rows: iterable of (ts, direction, track_id, session_id)."""
    rows = list(rows)
    if not rows:
        return
    with connect(path) as conn:
        conn.executemany(
            "INSERT INTO events (ts, direction, track_id, session_id) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )

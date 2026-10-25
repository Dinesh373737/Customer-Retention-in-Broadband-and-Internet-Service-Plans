"""Small SQLite metadata store with a connection per operation."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..core.serialization import json_safe


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return uuid4().hex


class Store:
    def __init__(self, home: Path):
        self.home = Path(home).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "retention.sqlite3"
        self.lock = threading.RLock()
        for child in ["datasets", "models", "exports"]:
            (self.home / child).mkdir(exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(kind,id))")

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def put(self, kind: str, identifier: str, payload: dict):
        data = json_safe(payload)
        data.setdefault("created_at", utcnow())
        with self.lock, self.connection() as connection:
            connection.execute("INSERT INTO records(kind,id,payload,created_at) VALUES(?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload", (kind, identifier, json.dumps(data, allow_nan=False), data["created_at"]))
        return data

    def get(self, kind: str, identifier: str):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, kind: str, identifier: str, **changes):
        with self.lock:
            payload = self.get(kind, identifier)
            if payload is None:
                raise KeyError(f"Unknown {kind}: {identifier}")
            payload.update(changes)
            payload["updated_at"] = utcnow()
            return self.put(kind, identifier, payload)

    def list(self, kind: str):
        with self.connection() as connection:
            rows = connection.execute("SELECT payload FROM records WHERE kind=? ORDER BY created_at DESC,id", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def recover_jobs(self):
        for job in self.list("job"):
            if job["status"] in {"queued", "running"}:
                message = "Training was interrupted by a backend restart. Start a new training job."
                self.update("job", job["id"], status="failed", error=message, message=message)
                if self.get("analysis", job["analysis_id"]):
                    self.update("analysis", job["analysis_id"], status="failed", error=message)

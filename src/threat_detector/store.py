"""Structured facts store — the non-RAG half of the hybrid design.

Module 3's key judgment call: **not everything belongs in the vector store.**
Stock prices, incident coordinates, travel dates, permit filing dates are
structured facts. Embedding them is worse than querying them — you'd be asking a
similarity search to answer "how many incidents within 500 m in the last 30
days," which it does badly. So those facts live here, in SQL, and the vector
index holds only unstructured text.

The store is **SQLite**: a single file, no server, in Python's standard library,
with real indexes and ``WHERE`` clauses. At capstone scale a spreadsheet would
work, but the metadata pre-filter *is* a query (entity + date range + geo box),
and SQLite answers that with an index instead of scanning every row — the right
shape that keeps working as ingest grows. ``export_csv`` is provided so the data
is still eyeball-able like a spreadsheet.

This store is what the retrieval pre-filter consults for facts, and what the
Geospatial/Public-Records agents write structured observations into.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

from .config import Config, load_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id     TEXT PRIMARY KEY,
    fact_type   TEXT NOT NULL,       -- permit | incident | docket | travel
    entity_id   TEXT NOT NULL,       -- which protectee/adversary this concerns
    source_id   TEXT NOT NULL,
    summary     TEXT NOT NULL,
    city        TEXT,
    lat         REAL,
    lon         REAL,
    event_date  TEXT,                -- ISO date; the pre-filter's date range column
    severity    REAL DEFAULT 0.0,
    reliability REAL DEFAULT 0.6
);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_date   ON facts(event_date);
CREATE INDEX IF NOT EXISTS idx_facts_type   ON facts(fact_type);
"""

_COLUMNS = ["fact_id", "fact_type", "entity_id", "source_id", "summary",
            "city", "lat", "lon", "event_date", "severity", "reliability"]


class FactsStore:
    """Thin wrapper over a SQLite database of structured facts."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # --- ingest ------------------------------------------------------------
    def reset(self) -> None:
        self.conn.execute("DELETE FROM facts")
        self.conn.commit()

    def add(self, rows: list[dict]) -> None:
        placeholders = ", ".join(f":{c}" for c in _COLUMNS)
        self.conn.executemany(
            f"INSERT OR REPLACE INTO facts ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            [{c: r.get(c) for c in _COLUMNS} for r in rows],
        )
        self.conn.commit()

    def seed_from_fixture(self, path: Path) -> "FactsStore":
        """Rebuild the store from a JSON seed (idempotent: reset + insert)."""
        rows = json.loads(Path(path).read_text()).get("facts", [])
        self.reset()
        self.add(rows)
        return self

    # --- the metadata pre-filter (as SQL) ----------------------------------
    def query(
        self,
        entity_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        near: tuple[float, float, float] | None = None,   # (lat, lon, radius_km)
        fact_type: str | None = None,
        min_reliability: float | None = None,
    ) -> list[dict]:
        """Return facts matching the structured pre-filter.

        ``entity_id`` is the load-bearing filter: it hard-excludes facts about a
        different person of the same name, which a similarity search cannot do
        reliably. Geographic filtering uses a bounding box (cheap, index-free on
        lat/lon) followed by an exact haversine radius check in Python.
        """
        clauses, params = [], []
        if entity_id:
            clauses.append("entity_id = ?"); params.append(entity_id)
        if since:
            clauses.append("(event_date IS NULL OR event_date >= ?)"); params.append(since)
        if until:
            clauses.append("(event_date IS NULL OR event_date <= ?)"); params.append(until)
        if fact_type:
            clauses.append("fact_type = ?"); params.append(fact_type)
        if min_reliability is not None:
            clauses.append("reliability >= ?"); params.append(min_reliability)
        if near:
            lat, lon, radius_km = near
            dlat = radius_km / 111.0
            dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
            clauses += ["lat BETWEEN ? AND ?", "lon BETWEEN ? AND ?"]
            params += [lat - dlat, lat + dlat, lon - dlon, lon + dlon]

        sql = "SELECT * FROM facts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY event_date DESC"
        rows = [dict(r) for r in self.conn.execute(sql, params).fetchall()]

        if near:
            lat, lon, radius_km = near
            rows = [r for r in rows
                    if r["lat"] is None
                    or _haversine_km(lat, lon, r["lat"], r["lon"]) <= radius_km]
        return rows

    def all(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM facts").fetchall()]

    # --- eyeball-ability ---------------------------------------------------
    def export_csv(self, path: str | Path) -> Path:
        """Dump the store to CSV so it can be opened like a spreadsheet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.all()
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c) for c in _COLUMNS})
        return path

    def close(self) -> None:
        self.conn.close()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def open_facts_store(cfg: Config | None = None, in_memory: bool = False) -> FactsStore:
    """Open the configured facts store, seeded from the fixture.

    ``in_memory=True`` gives a hermetic store for tests; otherwise it builds the
    on-disk ``.db`` file so the analyst can inspect it.
    """
    cfg = cfg or load_config()
    seed = cfg.retrieval_path("facts")
    path = ":memory:" if in_memory else cfg.retrieval_path("facts_db")
    return FactsStore(path).seed_from_fixture(seed)


__all__ = ["FactsStore", "open_facts_store"]

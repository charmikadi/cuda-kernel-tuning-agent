"""Every experiment is logged to SQLite so runs can be compared and reproduced."""
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "results" / "experiments.db"


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT, iteration INTEGER, kernel INTEGER, tile INTEGER, coarsen INTEGER,
        status TEXT, gflops REAL, pct_of_cublas REAL, compile_s REAL,
        reasoning TEXT, raw TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    return conn


def log(run_id: str, iteration: int, result: dict, reasoning: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO experiments (run_id, iteration, kernel, tile, coarsen, status, gflops,"
            " pct_of_cublas, compile_s, reasoning, raw) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, iteration, result.get("kernel"), result.get("tile"), result.get("coarsen"),
             result.get("status"), result.get("gflops"), result.get("pct_of_cublas"),
             result.get("compile_s"), reasoning, json.dumps(result)))


def all_results(run_id: str | None = None) -> list[dict]:
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM experiments" + (" WHERE run_id = ?" if run_id else "")
        return [dict(r) for r in conn.execute(query, (run_id,) if run_id else ())]

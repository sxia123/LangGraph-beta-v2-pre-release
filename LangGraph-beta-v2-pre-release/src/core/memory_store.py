import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "memory.db"))

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        db_file = os.path.abspath(DB_PATH)
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        _conn = sqlite3.connect(db_file, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _ensure_schema(_conn)
    return _conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                event TEXT,
                input TEXT,
                result TEXT,
                metadata TEXT,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                pipeline TEXT,
                input TEXT,
                status TEXT,
                final_answer TEXT,
                started_at TEXT,
                finished_at TEXT,
                duration_s REAL,
                memory_count INTEGER DEFAULT 0,
                metadata TEXT
            )
            """
        )
        # Migrate pre-existing memories tables that lack the run_id column.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "run_id" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN run_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_run_id ON memories (run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs (started_at)")


def save_memory(entry: Dict[str, Any]) -> str:
    """Save a memory entry to SQLite and return the row id.

    Expected keys: event, input, result, timestamp, plus any other metadata.
    Pass "run_id" to group the memory under a specific pipeline run.
    """
    conn = _get_conn()
    mem_id = entry.get("id") or str(uuid.uuid4())
    run_id = entry.get("run_id")
    event = entry.get("event")
    input_val = json.dumps(entry.get("input"), ensure_ascii=False) if entry.get("input") is not None else None
    result_val = json.dumps(entry.get("result"), ensure_ascii=False) if entry.get("result") is not None else None
    metadata = {}
    if isinstance(entry.get("metadata"), dict):
        metadata.update(entry["metadata"])
    for k, v in entry.items():
        if k not in ("id", "run_id", "event", "input", "result", "timestamp", "metadata"):
            metadata[k] = v
    metadata_val = json.dumps(metadata, ensure_ascii=False) if metadata else None

    timestamp = entry.get("timestamp")

    with _lock:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, run_id, event, input, result, metadata, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mem_id, run_id, event, input_val, result_val, metadata_val, timestamp),
            )
    return mem_id


def fetch_memories(limit: int = 50, run_id: Optional[str] = None) -> list:
    """Fetch the most recent memories, optionally scoped to a single run."""
    conn = _get_conn()
    if run_id:
        cur = conn.execute(
            "SELECT id, run_id, event, input, result, metadata, timestamp FROM memories WHERE run_id = ? ORDER BY timestamp ASC LIMIT ?",
            (run_id, limit),
        )
    else:
        cur = conn.execute(
            "SELECT id, run_id, event, input, result, metadata, timestamp FROM memories ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
    rows = cur.fetchall()
    out = []
    for r in rows:
        try:
            inp = json.loads(r[3]) if r[3] else None
        except Exception:
            inp = r[3]
        try:
            res = json.loads(r[4]) if r[4] else None
        except Exception:
            res = r[4]
        try:
            meta = json.loads(r[5]) if r[5] else None
        except Exception:
            meta = r[5]
        out.append({"id": r[0], "run_id": r[1], "event": r[2], "input": inp, "result": res, "metadata": meta, "timestamp": r[6]})
    return out


def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        meta = json.loads(row["metadata"]) if row["metadata"] else None
    except Exception:
        meta = row["metadata"]
    return {
        "run_id": row["run_id"],
        "pipeline": row["pipeline"],
        "input": row["input"],
        "status": row["status"],
        "final_answer": row["final_answer"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_s": row["duration_s"],
        "memory_count": row["memory_count"],
        "metadata": meta,
    }


# ---------------------------------------------------------------------- #
# Run tracking: one row per pipeline execution, memories grouped by run_id
# ---------------------------------------------------------------------- #
def start_run(pipeline: str, user_input: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Open a new run record and return its run_id."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    with _lock:
        with conn:
            conn.execute(
                "INSERT INTO runs (run_id, pipeline, input, status, started_at, metadata) VALUES (?, ?, ?, 'running', ?, ?)",
                (
                    run_id,
                    pipeline,
                    user_input,
                    started_at,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
    return run_id


def finish_run(
    run_id: str,
    status: str = "completed",
    final_answer: Optional[str] = None,
    started_at: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Close a run record: set status, final answer, duration, and memory count."""
    conn = _get_conn()
    finished_at = datetime.now().isoformat(timespec="seconds")
    if not started_at:
        row = conn.execute("SELECT started_at FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        started_at = row[0] if row else None
    duration = None
    if started_at:
        try:
            duration = round(time.time() - datetime.fromisoformat(started_at).timestamp(), 3)
        except Exception:
            duration = None
    count_row = conn.execute("SELECT COUNT(*) FROM memories WHERE run_id = ?", (run_id,)).fetchone()
    memory_count = int(count_row[0]) if count_row else 0
    with _lock:
        with conn:
            conn.execute(
                "UPDATE runs SET status = ?, final_answer = ?, finished_at = ?, duration_s = ?, memory_count = ? WHERE run_id = ?",
                (status, final_answer, finished_at, duration, memory_count, run_id),
            )


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single run record by id (None if not found)."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def list_runs(limit: int = 50) -> list:
    """List the most recent run records (newest first)."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_run(r) for r in rows]


def get_run_memories(run_id: str) -> list:
    """Fetch every memory row stored under a run (chronological order)."""
    return fetch_memories(limit=10_000, run_id=run_id)


def get_tool_checkpoints(
    run_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    checkpoint_type: Optional[str] = None,
    limit: int = 50,
) -> list:
    """Fetch tool execution checkpoints, optionally filtered by run_id, tool_name, or checkpoint_type."""
    conn = _get_conn()
    clauses = ["(event LIKE 'checkpoint_before_%' OR event LIKE 'checkpoint_after_%' OR event LIKE 'checkpoint_error_%' OR metadata LIKE '%checkpoint_type%')"]
    params: list[Any] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if tool_name:
        clauses.append("(event LIKE ? OR metadata LIKE ?)")
        params.append(f"%{tool_name}%")
        params.append(f"%{tool_name}%")
    if checkpoint_type:
        clauses.append("metadata LIKE ?")
        params.append(f"%{checkpoint_type}%")

    where_sql = " AND ".join(clauses)
    params.append(limit)
    cur = conn.execute(
        f"SELECT id, run_id, event, input, result, metadata, timestamp FROM memories WHERE {where_sql} ORDER BY timestamp ASC LIMIT ?",
        tuple(params),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        try:
            inp = json.loads(r[3]) if r[3] else None
        except Exception:
            inp = r[3]
        try:
            res = json.loads(r[4]) if r[4] else None
        except Exception:
            res = r[4]
        try:
            meta = json.loads(r[5]) if r[5] else None
        except Exception:
            meta = r[5]
        out.append(
            {
                "id": r[0],
                "run_id": r[1],
                "event": r[2],
                "input": inp,
                "result": res,
                "metadata": meta,
                "timestamp": r[6],
            }
        )
    return out



def search_memories(query: str, limit: int = 50) -> list:
    """Search memories by keyword match across event, input, result, or metadata."""
    if not query or not query.strip():
        return fetch_memories(limit=limit)

    conn = _get_conn()
    pattern = f"%{query.strip()}%"
    cur = conn.execute(
        """
        SELECT id, run_id, event, input, result, metadata, timestamp
        FROM memories
        WHERE event LIKE ? OR input LIKE ? OR result LIKE ? OR metadata LIKE ?
        ORDER BY timestamp DESC LIMIT ?
        """,
        (pattern, pattern, pattern, pattern, limit),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        try:
            inp = json.loads(r[3]) if r[3] else None
        except Exception:
            inp = r[3]
        try:
            res = json.loads(r[4]) if r[4] else None
        except Exception:
            res = r[4]
        try:
            meta = json.loads(r[5]) if r[5] else None
        except Exception:
            meta = r[5]
        out.append(
            {
                "id": r[0],
                "run_id": r[1],
                "event": r[2],
                "input": inp,
                "result": res,
                "metadata": meta,
                "timestamp": r[6],
            }
        )
    return out


def delete_memory(memory_id: str) -> bool:
    """Delete a single memory item by id."""
    conn = _get_conn()
    with _lock:
        with conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0


def clear_memories() -> int:
    """Clear all stored memories and reset run counts."""
    conn = _get_conn()
    with _lock:
        with conn:
            cur = conn.execute("DELETE FROM memories")
            deleted = cur.rowcount
            conn.execute("UPDATE runs SET memory_count = 0")
            return deleted


def delete_run(run_id: str) -> bool:
    """Delete a run record and all associated memories."""
    conn = _get_conn()
    with _lock:
        with conn:
            conn.execute("DELETE FROM memories WHERE run_id = ?", (run_id,))
            cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            return cur.rowcount > 0


def get_memory_by_id(memory_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single memory or checkpoint item by id."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id, run_id, event, input, result, metadata, timestamp FROM memories WHERE id = ?",
        (memory_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        inp = json.loads(row[3]) if row[3] else None
    except Exception:
        inp = row[3]
    try:
        res = json.loads(row[4]) if row[4] else None
    except Exception:
        res = row[4]
    try:
        meta = json.loads(row[5]) if row[5] else None
    except Exception:
        meta = row[5]
    return {
        "id": row[0],
        "run_id": row[1],
        "event": row[2],
        "input": inp,
        "result": res,
        "metadata": meta,
        "timestamp": row[6],
    }


def save_file_checkpoint(
    file_path: str,
    content: Optional[str] = None,
    exists: bool = False,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves a file modification checkpoint to SQLite memory."""
    abs_path = os.path.abspath(file_path)
    norm_path = abs_path.replace("\\", "/")
    meta = dict(metadata or {})
    checkpoint_type = meta.get("checkpoint_type", "pre_file_change")
    event_name = f"{checkpoint_type}:{os.path.basename(file_path)}"

    entry = {
        "run_id": run_id,
        "event": event_name,
        "input": {
            "file_path": abs_path,
            "normalized_path": norm_path,
            "exists": exists,
            "content": content,
        },
        "result": {
            "status": "recorded",
            "file_path": abs_path,
            "normalized_path": norm_path,
        },
        "metadata": meta,
    }
    return save_memory(entry)


def get_file_checkpoints(
    file_path: Optional[str] = None, run_id: Optional[str] = None, limit: int = 50
) -> list:
    """Retrieve file checkpoints from SQLite memory, optionally filtered by file_path or run_id."""
    conn = _get_conn()
    clauses = ["event LIKE '%file_change%'"]
    params: list = []

    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)

    if file_path:
        abs_p = os.path.abspath(file_path)
        norm_p = abs_p.replace("\\", "/")
        esc_p = abs_p.replace("\\", "\\\\")
        base_p = os.path.basename(abs_p)
        clauses.append(
            "("
            "INSTR(COALESCE(input, ''), ?) > 0 OR INSTR(COALESCE(metadata, ''), ?) > 0 OR "
            "INSTR(COALESCE(input, ''), ?) > 0 OR INSTR(COALESCE(metadata, ''), ?) > 0 OR "
            "INSTR(COALESCE(input, ''), ?) > 0 OR INSTR(COALESCE(metadata, ''), ?) > 0 OR "
            "INSTR(COALESCE(event, ''), ?) > 0"
            ")"
        )
        params.extend([abs_p, abs_p, norm_p, norm_p, esc_p, esc_p, base_p])

    where_sql = " AND ".join(clauses)
    params.append(limit)
    cur = conn.execute(
        f"SELECT id, run_id, event, input, result, metadata, timestamp FROM memories WHERE {where_sql} ORDER BY timestamp DESC LIMIT ?",
        tuple(params),
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        try:
            inp = json.loads(r[3]) if r[3] else None
        except Exception:
            inp = r[3]
        try:
            res = json.loads(r[4]) if r[4] else None
        except Exception:
            res = r[4]
        try:
            meta = json.loads(r[5]) if r[5] else None
        except Exception:
            meta = r[5]
        out.append(
            {
                "id": r[0],
                "run_id": r[1],
                "event": r[2],
                "input": inp,
                "result": res,
                "metadata": meta,
                "timestamp": r[6],
            }
        )
    return out


def get_memory_summary() -> Dict[str, Any]:
    """Retrieve memory statistics for persistent storage overview."""
    conn = _get_conn()
    mem_count_row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    run_count_row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    last_mem_row = conn.execute(
        "SELECT timestamp FROM memories ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    total_memories = int(mem_count_row[0]) if mem_count_row else 0
    total_runs = int(run_count_row[0]) if run_count_row else 0
    last_saved = last_mem_row[0] if last_mem_row else None

    return {
        "total_memories": total_memories,
        "total_runs": total_runs,
        "last_saved": last_saved,
        "db_path": os.path.abspath(DB_PATH),
    }




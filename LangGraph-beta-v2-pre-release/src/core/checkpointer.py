"""LangGraph checkpointer integration and pre-file-change snapshot management.

Provides LangGraph StateGraph checkpointer access along with automatic file-level
checkpointing before any file mutation (write, edit, delete, patch, or REPL execution).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.core.memory_store import (
    get_file_checkpoints as db_get_file_checkpoints,
)
from src.core.memory_store import (
    get_memory_by_id,
)
from src.core.memory_store import (
    save_file_checkpoint as db_save_file_checkpoint,
)

_global_checkpointer: Optional[BaseCheckpointSaver] = None


def get_default_checkpointer() -> BaseCheckpointSaver:
    """Returns a singleton MemorySaver checkpointer for LangGraph graphs."""
    global _global_checkpointer
    if _global_checkpointer is None:
        _global_checkpointer = MemorySaver()
    return _global_checkpointer


def reset_default_checkpointer() -> BaseCheckpointSaver:
    """Resets and returns a fresh MemorySaver instance (useful for test isolation)."""
    global _global_checkpointer
    _global_checkpointer = MemorySaver()
    return _global_checkpointer


def compute_file_sha256(file_path: str) -> Optional[str]:
    """Computes SHA256 hex digest of a file if it exists."""
    if not os.path.isfile(file_path):
        return None
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def checkpoint_before_file_change(
    file_path: str,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Captures a full snapshot of a file before any change or write operation.

    Stores the file path, existence status, original content, SHA256 checksum,
    file size, and metadata in SQLite memory for verification and rollback.
    """
    abs_path = os.path.abspath(file_path)
    exists = os.path.exists(abs_path)
    content: Optional[str] = None
    size: int = 0
    sha256_hash: Optional[str] = None

    if exists and os.path.isfile(abs_path):
        try:
            size = os.path.getsize(abs_path)
            sha256_hash = compute_file_sha256(abs_path)
            # Try reading as text; fallback to hex if binary
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(abs_path, "rb") as f:
                    content = f.read().hex()
        except Exception as err:
            content = f"<error reading content: {err}>"

    meta = dict(metadata or {})
    meta.update(
        {
            "checkpoint_type": "pre_file_change",
            "file_path": abs_path,
            "exists_before": exists,
            "size_before": size,
            "sha256_before": sha256_hash,
        }
    )

    checkpoint_id = db_save_file_checkpoint(
        file_path=abs_path,
        content=content,
        exists=exists,
        run_id=run_id,
        metadata=meta,
    )

    return {
        "ok": True,
        "checkpoint_id": checkpoint_id,
        "file_path": abs_path,
        "exists_before": exists,
        "size_before": size,
        "sha256_before": sha256_hash,
    }


def checkpoint_after_file_change(
    file_path: str,
    pre_checkpoint_id: str,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Records a post-change checkpoint after a file mutation."""
    abs_path = os.path.abspath(file_path)
    exists = os.path.exists(abs_path)
    size = os.path.getsize(abs_path) if exists and os.path.isfile(abs_path) else 0
    sha256_hash = compute_file_sha256(abs_path) if exists else None

    meta = dict(metadata or {})
    meta.update(
        {
            "checkpoint_type": "post_file_change",
            "file_path": abs_path,
            "pre_checkpoint_id": pre_checkpoint_id,
            "exists_after": exists,
            "size_after": size,
            "sha256_after": sha256_hash,
        }
    )

    checkpoint_id = db_save_file_checkpoint(
        file_path=abs_path,
        content=None,
        exists=exists,
        run_id=run_id,
        metadata=meta,
    )

    return {
        "ok": True,
        "checkpoint_id": checkpoint_id,
        "pre_checkpoint_id": pre_checkpoint_id,
        "file_path": abs_path,
        "exists_after": exists,
        "size_after": size,
        "sha256_after": sha256_hash,
    }


def rollback_file_checkpoint(checkpoint_id: str) -> Dict[str, Any]:
    """Rolls back a file to the exact state saved in a pre-file-change checkpoint."""
    record = get_memory_by_id(checkpoint_id)
    if not record:
        return {"ok": False, "message": f"Checkpoint '{checkpoint_id}' not found."}

    metadata = record.get("metadata") or {}
    file_path = metadata.get("file_path") or record.get("input", {}).get("file_path")
    if not file_path:
        return {"ok": False, "message": "Checkpoint record does not contain target file_path."}

    exists_before = metadata.get("exists_before", False)
    content_payload = record.get("input", {}).get("content")

    try:
        if not exists_before:
            # The file did not exist before the change -> remove it if it exists now
            if os.path.exists(file_path):
                os.remove(file_path)
            return {
                "ok": True,
                "file_path": file_path,
                "action": "removed_new_file",
                "message": f"Rolled back file '{file_path}' (removed created file).",
            }
        else:
            # Restore previous content
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            if content_payload is not None:
                # Check if it was hex-encoded binary
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content_payload)
                except Exception:
                    with open(file_path, "wb") as f:
                        f.write(bytes.fromhex(content_payload))
            return {
                "ok": True,
                "file_path": file_path,
                "action": "restored_content",
                "message": f"Rolled back file '{file_path}' to pre-change snapshot.",
            }
    except Exception as err:
        return {"ok": False, "file_path": file_path, "message": f"Rollback failed: {err}"}


def get_file_checkpoints(
    file_path: Optional[str] = None, run_id: Optional[str] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """Retrieves file checkpoints filtered by path or run_id."""
    return db_get_file_checkpoints(file_path=file_path, run_id=run_id, limit=limit)

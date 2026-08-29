"""Imperative GitHub tool: clone, pull, commit, and push repositories via the git CLI.

This is a plain-Python tool (no LLM function calling required). Pipeline nodes
call these functions directly, exactly like `llm_client.search_web(...)`, and
inject the result into the prompt.

Security notes:
- All git invocations use argument lists (never a shell), so no shell
  injection is possible from repo URLs, paths, or commit messages.
- Every call has a hard wall-clock timeout so a hung network operation
  (e.g. an unreachable remote) can never block the pipeline.
- Auth is handled by the environment: use a token in the URL
  (https://<token>@github.com/user/repo.git) or a configured credential
  helper / SSH key. No credentials are stored here.

Usage from a pipeline node:

    from src.core.github_tool import pull_repo, commit_and_push, format_result

    result = pull_repo("/path/to/repo")
    prompt = f"...\\nGitHub Pull Output:\\n{format_result(result)}\\n..."
"""

import os
import subprocess
from typing import Any, Dict, Optional

GIT_TIMEOUT = int(os.getenv("GIT_TOOL_TIMEOUT", "120"))


def _run_git(args: list, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Run a git command safely and return a structured result."""
    cmd = ["git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout or GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "message": f"git {' '.join(args)} timed out after {timeout or GIT_TIMEOUT}s",
            "returncode": None,
        }
    except FileNotFoundError:
        return {"ok": False, "message": "git executable not found on PATH", "returncode": None}

    output = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    return {
        "ok": ok,
        "message": output.strip() or ("success" if ok else "failed"),
        "returncode": proc.returncode,
    }


def _is_git_repo(repo_dir: str) -> bool:
    return os.path.isdir(os.path.join(repo_dir, ".git"))


def clone_repo(repo_url: str, dest_dir: str, branch: Optional[str] = None) -> Dict[str, Any]:
    """Clone a GitHub repository (or any git URL) into dest_dir."""
    if not repo_url or not repo_url.strip():
        return {"ok": False, "message": "repo_url is required", "returncode": None}
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        return {
            "ok": False,
            "message": f"Destination {dest_dir} already exists and is not empty",
            "returncode": None,
        }

    args = ["clone"]
    if branch:
        args += ["--branch", branch, "--"]
    args += [repo_url.strip(), dest_dir]
    return _run_git(args)


def pull_repo(repo_dir: str, branch: Optional[str] = None) -> Dict[str, Any]:
    """Pull the latest changes for a local repository."""
    if not os.path.isdir(repo_dir):
        return {"ok": False, "message": f"Repository directory not found: {repo_dir}", "returncode": None}
    if not _is_git_repo(repo_dir):
        return {"ok": False, "message": f"Not a git repository: {repo_dir}", "returncode": None}

    args = ["pull", "--ff-only"]
    if branch:
        args += ["--", branch]
    return _run_git(args, cwd=repo_dir)


def push_repo(repo_dir: str, remote: str = "origin", branch: Optional[str] = None) -> Dict[str, Any]:
    """Push local commits to a remote."""
    if not os.path.isdir(repo_dir):
        return {"ok": False, "message": f"Repository directory not found: {repo_dir}", "returncode": None}
    if not _is_git_repo(repo_dir):
        return {"ok": False, "message": f"Not a git repository: {repo_dir}", "returncode": None}

    args = ["push"]
    if branch:
        args += [remote, branch]
    else:
        args += [remote]
    return _run_git(args, cwd=repo_dir)


def commit_and_push(
    repo_dir: str,
    message: str,
    remote: str = "origin",
    branch: Optional[str] = None,
    author_name: str = "LangGraph Agent",
    author_email: str = "agent@localhost",
) -> Dict[str, Any]:
    """Stage all changes, commit them, and push to the remote.

    Author identity is passed per-invocation (-c user.name=...) so the tool
    works even when no global git identity is configured.
    """
    if not os.path.isdir(repo_dir):
        return {"ok": False, "message": f"Repository directory not found: {repo_dir}", "returncode": None}
    if not _is_git_repo(repo_dir):
        return {"ok": False, "message": f"Not a git repository: {repo_dir}", "returncode": None}
    if not message or not message.strip():
        return {"ok": False, "message": "commit message is required", "returncode": None}

    add = _run_git(["add", "-A"], cwd=repo_dir, timeout=30)
    if not add["ok"]:
        return add

    commit = _run_git(
        [
            "-c", f"user.name={author_name}",
            "-c", f"user.email={author_email}",
            "commit", "-m", message.strip(),
        ],
        cwd=repo_dir,
        timeout=60,
    )
    if not commit["ok"]:
        if "nothing to commit" in commit["message"].lower():
            return {
                "ok": True,
                "message": "No changes to commit; nothing to push.",
                "returncode": commit["returncode"],
            }
        return commit

    push = push_repo(repo_dir, remote=remote, branch=branch)
    if not push["ok"]:
        return {
            "ok": False,
            "message": f"Commit succeeded but push failed: {push['message']}",
            "returncode": push["returncode"],
        }
    return {
        "ok": True,
        "message": f"Committed and pushed.\n{commit['message']}\n{push['message']}",
        "returncode": 0,
    }


def get_repo_status(repo_dir: str) -> Dict[str, Any]:
    """Return a short status summary (branch, ahead/behind, dirty files)."""
    if not os.path.isdir(repo_dir) or not _is_git_repo(repo_dir):
        return {"ok": False, "message": f"Not a git repository: {repo_dir}", "returncode": None}
    return _run_git(["status", "--short", "--branch"], cwd=repo_dir, timeout=30)


def format_result(result: Dict[str, Any]) -> str:
    """Format a tool result as a compact string for prompt injection."""
    if not result:
        return "No result."
    status = "OK" if result.get("ok") else "FAILED"
    return f"[{status}] {result.get('message', 'No message')}"

"""Subprocess helpers for QBlade-facing workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import signal
import subprocess
from typing import Any


def isolated_process_kwargs() -> dict[str, object]:
    """Return platform-specific kwargs that isolate a launched subprocess."""
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creationflags} if creationflags else {}
    return {"start_new_session": True}


def launch_isolated_process(
    command: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    launcher: Callable[..., Any] | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Launch a subprocess in an isolated session/process group when supported."""
    popen = subprocess.Popen if launcher is None else launcher
    kwargs = isolated_process_kwargs()
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if env is not None:
        kwargs["env"] = env
    kwargs.update(extra_kwargs)
    process = popen([str(part) for part in command], **kwargs)
    try:
        setattr(process, "_vorlap_isolated_process", True)
    except Exception:
        pass
    return process


def _process_returncode(process: Any) -> int | None:
    poll = getattr(process, "poll", None)
    if poll is None:
        return None
    try:
        return poll()
    except Exception:
        return None


def _terminate(process: Any, *, sig: int) -> None:
    pid = getattr(process, "pid", None)
    isolated = bool(getattr(process, "_vorlap_isolated_process", False))
    if os.name != "nt" and isolated and isinstance(pid, int) and pid > 0:
        try:
            os.killpg(os.getpgid(pid), sig)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass
    method_name = "kill" if sig == getattr(signal, "SIGKILL", sig) else "terminate"
    method = getattr(process, method_name, None)
    if method is not None:
        try:
            method()
        except ProcessLookupError:
            pass


def terminate_process_tree(process: Any, *, timeout_s: float = 10.0) -> int | None:
    """Terminate a process and its process group when launched by this module."""
    returncode = _process_returncode(process)
    if returncode is not None:
        return returncode
    _terminate(process, sig=signal.SIGTERM)
    wait = getattr(process, "wait", None)
    if wait is None:
        return _process_returncode(process)
    try:
        return wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _terminate(process, sig=getattr(signal, "SIGKILL", signal.SIGTERM))
        return wait(timeout=timeout_s)

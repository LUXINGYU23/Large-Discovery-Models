"""Single-controller run lock (design 5.3): an exclusive, non-blocking ``flock`` next to the store."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class RunLockHeld(RuntimeError):
    pass


class RunLock:
    def __init__(self, path):
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RunLockHeld(f"another controller holds {self.path}") from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()

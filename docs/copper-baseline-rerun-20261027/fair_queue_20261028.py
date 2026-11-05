"""FIFO admission for the qualified operational scheduler; no scientific changes."""
from collections import deque
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import threading
import time
from tasks.copper_inversion.experiments.v2.parallel_runtime import JobQueue as OriginalQueue


def prepare(root):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    root.chmod(0o777)
    (root / 'admission.lock').touch(exist_ok=True)
    (root / 'admission.lock').chmod(0o666)
    for channel in ('solver', 'interface'):
        (root / channel).mkdir(exist_ok=True)
        (root / channel).chmod(0o777)


class JobQueue(OriginalQueue):
    def __init__(self, root, slots, local_slots):
        super().__init__(root, slots, local_slots)
        self._condition = threading.Condition()
        self._waiting = deque()
        self._active = 0
        self._limit = local_slots

    @contextmanager
    def _local_fifo(self):
        token = object()
        admitted = False
        with self._condition:
            self._waiting.append(token)
            try:
                while self._waiting[0] is not token or self._active >= self._limit:
                    self.check()
                    self._condition.wait(.1)
                self.check()
                self._waiting.popleft()
                self._active += 1
                admitted = True
                self._condition.notify_all()
            finally:
                if not admitted:
                    self._waiting.remove(token)
                    self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @contextmanager
    def acquire(self, *, interface=False):
        started = time.monotonic()
        with self._local_fifo():
            reserved = getattr(self, 'interface_slots', 0)
            indices = range(reserved) if interface and reserved else range(reserved, self.slots)
            channel = self.root / ('interface' if interface and reserved else 'solver')
            mutex = os.open(self.root / 'admission.lock', os.O_RDWR)
            ticket_fd = slot_fd = None
            ticket = None
            try:
                # Registration and selection share one mutex. Monotonic timestamps
                # are machine-wide; serialization prevents equal ticket numbers.
                fcntl.flock(mutex, fcntl.LOCK_EX)
                try:
                    self.check()
                    ticket = channel / f'{time.monotonic_ns():020d}-{os.getpid()}-{threading.get_ident()}'
                    ticket_fd = os.open(ticket, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666)
                    os.fchmod(ticket_fd, 0o666)
                    fcntl.flock(ticket_fd, fcntl.LOCK_EX)
                finally:
                    fcntl.flock(mutex, fcntl.LOCK_UN)
                while slot_fd is None:
                    self.check()
                    fcntl.flock(mutex, fcntl.LOCK_EX)
                    try:
                        # A waiting process owns its ticket lock. Death releases it;
                        # stale tickets cannot indefinitely block the next waiter.
                        head = None
                        for path in sorted(channel.iterdir()):
                            if path == ticket:
                                head = path
                                break
                            probe = os.open(path, os.O_RDWR)
                            try:
                                try:
                                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    head = path
                                    break
                                else:
                                    path.unlink()
                            finally:
                                os.close(probe)
                        if head == ticket:
                            for index in indices:
                                fd = os.open(self.root / f'{index:03}.lock', os.O_RDWR)
                                try:
                                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    os.close(fd)
                                else:
                                    slot_fd = fd
                                    ticket.unlink()
                                    break
                    finally:
                        fcntl.flock(mutex, fcntl.LOCK_UN)
                    if slot_fd is None:
                        time.sleep(.02)
                self.check()
                yield time.monotonic() - started
            except BaseException as exc:
                from tasks.copper_inversion.core.execution import EvaluationPaused
                if isinstance(exc, EvaluationPaused):
                    self.pause(exc)
                raise
            finally:
                if slot_fd is not None:
                    os.close(slot_fd)
                if ticket_fd is not None:
                    fcntl.flock(mutex, fcntl.LOCK_EX)
                    try:
                        ticket.unlink(missing_ok=True)
                        os.close(ticket_fd)
                    finally:
                        fcntl.flock(mutex, fcntl.LOCK_UN)
                os.close(mutex)

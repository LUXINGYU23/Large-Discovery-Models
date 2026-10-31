"""Bounded handoff from synchronous worker callbacks to the Host writer."""

from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Condition, get_ident


class HostDispatcher:
    def __init__(self):
        self.owner = get_ident()
        self.messages = Queue(maxsize=64)
        self.lifecycle = Condition()
        self.active = False

    def call(self, operation):
        if get_ident() == self.owner:
            return operation()
        answer = Future()
        with self.lifecycle:
            while self.active:
                try:
                    self.messages.put_nowait((operation, answer))
                    break
                except Full:
                    self.lifecycle.wait()
            else:
                raise RuntimeError("worker callback requires an active Host message pump")
        return answer.result()

    def run(self, operation):
        if get_ident() != self.owner or self.active:
            raise RuntimeError("only the Host may start a non-overlapping callback stage")
        with self.lifecycle:
            self.active = True
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="t3-stage") as pool:
                result = pool.submit(operation)
                interruption = None
                while not result.done() or not self.messages.empty():
                    answer = None
                    try:
                        callback, answer = self.messages.get(timeout=.05)
                        with self.lifecycle:
                            self.lifecycle.notify_all()
                        if interruption:
                            answer.set_exception(interruption)
                            continue
                        try:
                            answer.set_result(callback())
                        except BaseException as exc:
                            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                interruption = exc
                            answer.set_exception(exc)
                    except Empty:
                        continue
                    except (KeyboardInterrupt, SystemExit) as exc:
                        interruption = exc
                        if answer is not None and not answer.done():
                            answer.set_exception(exc)
                if interruption:
                    raise interruption
                return result.result()
        finally:
            with self.lifecycle:
                self.active = False
                while not self.messages.empty():
                    _, answer = self.messages.get_nowait()
                    answer.set_exception(RuntimeError("callback arrived after its Host stage ended"))
                self.lifecycle.notify_all()

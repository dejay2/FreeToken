import queue
import multiprocessing as mp
from types import SimpleNamespace

import pytest

from freetoken.server import launch


class Reports(queue.Queue):
    def close(self):
        pass

    def join_thread(self):
        pass


def test_fatal_scheduler_reports_before_exit_and_never_runs_cuda_shutdown(monkeypatch):
    reports = Reports()
    shutdown = []

    def fail():
        raise RuntimeError('CUDA illegal address')

    def exit_process(code):
        assert reports.get_nowait()[1].endswith('RuntimeError: CUDA illegal address')
        raise SystemExit(code)

    monkeypatch.setattr(launch.os, '_exit', exit_process)
    scheduler = SimpleNamespace(run_forever=fail, shutdown=lambda: shutdown.append(True))
    args = SimpleNamespace(tp_info=SimpleNamespace(rank=0, is_primary=lambda: True))
    with pytest.raises(SystemExit) as error:
        launch._serve_scheduler(scheduler, reports, args)
    assert error.value.code == 1
    assert shutdown == []


def test_keyboard_interrupt_keeps_the_graceful_shutdown_path():
    shutdown = []

    def interrupt():
        raise KeyboardInterrupt

    scheduler = SimpleNamespace(run_forever=interrupt, shutdown=lambda: shutdown.append(True))
    reports = Reports()
    args = SimpleNamespace(tp_info=SimpleNamespace(rank=0, is_primary=lambda: True))
    launch._serve_scheduler(scheduler, reports, args)
    assert shutdown == [True]
    assert reports.empty()


def _crashing_worker(reports, blocked_stderr):
    if blocked_stderr:
        import time
        launch.traceback.print_exception = lambda *a, **kw: time.sleep(60)
        launch.traceback.print_exc = lambda *a, **kw: time.sleep(60)
    def fail():
        raise RuntimeError("runtime IPC regression")

    scheduler = SimpleNamespace(run_forever=fail)
    args = SimpleNamespace(tp_info=SimpleNamespace(rank=0))
    launch._serve_scheduler(scheduler, reports, args)


@pytest.mark.parametrize("blocked_stderr", [False, True])
def test_actual_process_exit_flushes_the_multiprocessing_error_queue(blocked_stderr):
    context = mp.get_context("spawn")
    reports = context.Queue()
    worker = context.Process(target=_crashing_worker, args=(reports, blocked_stderr))
    worker.start()
    try:
        assert reports.get(timeout=15) == ("error", "runtime scheduler TP0: RuntimeError: runtime IPC regression")
        worker.join(timeout=3)
        assert worker.exitcode == 1
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=3)
        reports.close()

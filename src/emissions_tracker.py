import os
import threading
import time
import uuid
from typing import Callable

from codecarbon import OfflineEmissionsTracker


def flush_tracker_periodically(tracker, interval, stop_event, run_id) -> None:
    """
    Periodically flush tracker until stop_event is set.
    """
    while not stop_event.is_set():
        tracker.flush()
        time.sleep(interval)

    # Final flush on exit
    tracker.flush()
    print(f"[Thread] Tracker {run_id} stopped and flushed.")


def tracked_function(
    _command: Callable,
    _output_dir: str,
    _interval: int,
    _project_name: str,
    _args: tuple | None = None,
) -> None:
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)

    run_id = str(uuid.uuid4())
    output_file = f"emissions_{run_id}.csv"
    print(f"[Info] Output file: {output_file}")

    tracker = OfflineEmissionsTracker(
        country_iso_code="ITA",
        measure_power_secs=_interval,
        project_name=_project_name,
        output_dir=_output_dir,
        output_file=output_file,
        allow_multiple_runs=True,
        log_level="critical",
    )

    tracker.start()

    stop_event = threading.Event()
    flush_thread = threading.Thread(
        target=flush_tracker_periodically,
        args=(tracker, _interval, stop_event, run_id),
        daemon=True,
    )
    flush_thread.start()

    try:
        _command(*_args if _args else ())
    finally:
        tracker.stop()
        stop_event.set()
        flush_thread.join(timeout=5)

"""Shared file logging for the realtime viewers (mesh_viewer_segmented.py, live_preprocessing.py)
- routes every _debug() call across those files into one timestamped file per run under
testing/logs/ instead of flooding the terminal. A full run's debug output can then be handed over
by pointing at (or tailing) a file path instead of pasting thousands of lines into a chat prompt.

Call setup_logging() once, early, from the entry point (mesh_viewer_segmented.py's main(), before
the reader/worker threads start). Every module's logger.debug(...) then lands in that one file
regardless of import order or which file logging.getLogger("realtime") is called from -
logging.getLogger(name) returns the same singleton per name for the life of the process.
"""
import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

logger = logging.getLogger("realtime")
logger.setLevel(logging.DEBUG)


def setup_logging(tag="mesh_viewer_segmented"):
    """Creates testing/logs/ if needed and attaches a per-run timestamped FileHandler. Returns
    the log file path (printed once to stdout so there's always a pointer to where the full
    output landed)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{tag}_{timestamp}.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return log_path

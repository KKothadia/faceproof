import logging
import os
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure root logging once: console output plus a rotating file handler for
    auditability of pipeline runs (search/verification/blockchain events).
    Safe to call multiple times (idempotent) and never raises - if the log file
    can't be created (e.g. read-only filesystem), pipeline execution must not break.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "faceproof.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as e:
        logging.getLogger(__name__).warning("Could not set up file logging: %s", e)

    _CONFIGURED = True

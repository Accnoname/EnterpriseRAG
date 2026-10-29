import logging
import sys

def get_logger(name: str = "EnterpriseRAG") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Use utf-8 encoding on Windows to prevent UnicodeEncodeError
        handler = logging.StreamHandler(sys.stdout)
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except (AttributeError, io.UnsupportedOperation):
                pass
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

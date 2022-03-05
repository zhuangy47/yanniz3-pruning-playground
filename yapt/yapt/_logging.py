import logging
import os
from logging import config
from pathlib import Path
from typing import Iterable

import tqdm

from .utils import PathLike, resolve_trainer_dir


class TqdmStreamHandler(logging.Handler):
    """tqdm-friendly logging handler. Uses tqdm.write instead of print for logging."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            self.handleError(record)
            raise
        except:
            self.handleError(record)


def config_pylogger(
    dir_or_trainer=None,
    filename: str = None,
    verbose: bool = False,
    name_filters: Iterable[str] = ("PIL.PngImagePlugin",),
) -> logging.Logger:
    """Configure the Python logger.

    For each execution of the application, we'd like to create a unique log file.
    By default this file is named using the date and time of day, so that it can be sorted by recency.
    You can also name your filename or choose the log directory.
    """
    import time

    from pytorch_lightning import Trainer

    name_filters = set(name_filters)

    class NameFilter(logging.Filter):
        def filter(self, record):
            return not record.name in name_filters

    filename = filename or time.strftime("%Y.%m.%d-%H%M%S.log")
    output_dir: PathLike
    if dir_or_trainer is None:
        output_dir = "."
    elif isinstance(dir_or_trainer, Trainer):
        output_dir = resolve_trainer_dir(dir_or_trainer)
    else:
        output_dir = dir_or_trainer
    os.makedirs(output_dir, exist_ok=True)
    file_path = (Path(output_dir) / filename).as_posix()

    d = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(levelname)s %(name)s: " "%(message)s"},
            "detailed": {
                "format": "[%(asctime)-15s] "
                "%(levelname)7s %(name)s: "
                "%(message)s "
                "@%(filename)s:%(lineno)d"
            },
        },
        "handlers": {
            "console": {
                "()": TqdmStreamHandler,
                "level": "DEBUG" if verbose else "INFO",
                "formatter": "simple",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "filename": file_path,
                "mode": "w",
                "formatter": "detailed",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }
    config.dictConfig(d)

    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(NameFilter())
    root.info(f"Log file for this run: {file_path}")
    return root

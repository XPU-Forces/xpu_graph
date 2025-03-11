import logging
import sys
import time
import functools

logger = logging.getLogger("XPU_GRAPH")


def setup_logger(loglevel):
    # Skip if handlers already exist
    if logger.handlers:
        return

    fmt = logging.Formatter(
        fmt="[XPU_GRAPH]: %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    logger.propagate = False


def xpu_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()

        # Determine function name (including class name if applicable)
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            func_name = func.__name__

        logger.debug(f"{func_name} cost {end - start:.4f}s")
        return res

    return wrapper

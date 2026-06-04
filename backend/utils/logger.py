import time
import logging
from functools import wraps


logger = logging.getLogger("rag-agent")


def log_node_time(node_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            start = time.time()

            try:
                result = func(state, *args, **kwargs)
                cost = (time.time() - start) * 1000
                logger.info("%s success cost=%.2fms", node_name, cost)
                return result

            except Exception as e:
                cost = (time.time() - start) * 1000
                logger.exception("%s failed cost=%.2fms error=%s", node_name, cost, e)
                raise

        return wrapper

    return decorator
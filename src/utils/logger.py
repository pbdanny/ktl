from functools import wraps

from pyspark.sql import SparkSession
from typing import Optional

def logger(fn):
    '''
    Decorator to log function name and arguments
    param
        fn: function, function to be decorated
    return
        function, decorated function
    '''
    from datetime import datetime, timezone

    @wraps(fn)
    def inner(*args, **kwargs):
        called_at = datetime.now(timezone.utc)
        print(f">>> Running {fn.__name__!r} function. Logged at {called_at}")
        print(f">>> kwargs {kwargs}")
        to_execute = fn(*args, **kwargs)
        print(f">>> Function executed, elapsed {(datetime.now(timezone.utc) - called_at).total_seconds()} sec.")
        return to_execute
    return inner
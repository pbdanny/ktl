import json
from .logger import logger

@logger
def conf_reader(path):
    with open(path, "r") as f:
        mapper = json.loads(f.read())
    return mapper

@logger
def conf_writer(mapper, path):
    with open(path, "w") as f:
        json.dump(mapper, f, indent=4, sort_keys=False)
    return
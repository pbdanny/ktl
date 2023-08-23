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
        json.dump(mapper, f, indent=4, sort_keys=True)
    return        

@logger
def save(df, path, format="delta", mode="overwrite", overwriteSchema=None):
    if overwriteSchema:
        df.write.format(format).mode(mode).option("overwriteSchema", True).save(
            path)
    else:
        df.write.format(format).mode(mode).save(path)
    print(">>> Saved to {}".format(path))

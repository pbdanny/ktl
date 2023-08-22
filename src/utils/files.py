import json
from .logger import logger

@logger
def conf_reader(path):
    f = open(path, "r")
    mapper = json.loads(f.read())
    f.close()
    return mapper

@logger
def conf_writer(mapper, path):
    with open(path, "w") as f:
        json.dump(mapper, f)
    return        

@logger
def save(df, path, format="delta", mode="overwrite", overwriteSchema=None):
    if overwriteSchema:
        df.write.format(format).mode(mode).option("overwriteSchema", True).save(
            path)
    else:
        df.write.format(format).mode(mode).save(path)
    print(">>> Saved to {}".format(path))

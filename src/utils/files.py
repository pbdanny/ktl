import json
from .logger import logger

@logger
def save(df, path, format="delta", mode="overwrite", overwriteSchema=None):
    if overwriteSchema:
        df.write.format(format).mode(mode).option("overwriteSchema", True).save(
            path)
    else:
        df.write.format(format).mode(mode).save(path)
    print(">>> Saved to {}".format(path))

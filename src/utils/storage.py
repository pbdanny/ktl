from .logger import logger

@logger
def save_hive(conf_mapper, sf, name):
    (sf
     .write
     .mode("overwrite")
     .saveAsTable(conf_mapper["storage"]["hive"]["prefix"] + name)
    )
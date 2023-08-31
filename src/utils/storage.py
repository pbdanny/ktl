from .logger import logger

@logger
def save_hive(sf, conf_mapper, name):
    """
    params
    ----
    sf: SparkDataFrame
    conf_mapper: dict
    name: str
    
    return
    -----
    None
    """
    (sf
     .write
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(conf_mapper["storage"]["hive"]["prefix"] + name)
    )
    return
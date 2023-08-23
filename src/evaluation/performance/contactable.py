'''
Created on 2013-3-13
@summary: This module is used to evaluate the performance of contactable
@version: 1.0
@since: 2013-3-13
Dimention
    1. contactable
detail
    AZAY vs CIG
    Number of Lead -> Contactable -> Bought rate]

'''
from utils.logger import logger
import os
import sys
import pandas as pd
from pyspark.sql import functions as F
from pathlib import Path
import argparse
from datetime import datetime, timedelta

sys.path.append('../')


def gen(spark, dbfs_path, insurer, conf_mapper, fdbk_type):
    from utils import etl
    fdbk_df, filename = etl.get_df(
        dbfs_path, insurer, conf_mapper, fdbk_type=fdbk_type)
    cust_fdbck_hh = etl.ngc_to_household(
        spark, spark.createDataFrame(fdbk_df, schema=None))
    return cust_fdbck_hh


def lead_mapper_transform(spark, lead_mapper):
    '''
        transform the lead mapper for missing lot azay only
        :param: 
            spark
            lead_mapper
    '''
    mapper_path = lead_mapper["path"]
    lead_map = pd.read_csv(mapper_path)
    lead_map = spark.createDataFrame(lead_map)
    lead_map = lead_map.withColumn("Filename", F.expr(
        "substring(Filename, 1, length(Filename)-2)"))
    lead_map = lead_map.withColumn(
        "Filename", F.concat(F.col("Filename"), F.lit(".txt")))

    return lead_map


def main(spark, insurer: str, mapper_lead: bool = False):
    '''
        main function
        :param:
            spark
            insurer
    '''
    from src.utils import conf
    from etl import staging
    conf_mapper = conf.conf_reader("../config/etl.json")
    mnt_mapper = conf.conf_reader("../config/mnt.json")
    lead_mapper = conf.conf_reader("../config/lead_lot.json")

    abfss_prefix, dbfs_path = (
        mnt_mapper["abfss_prefix"], mnt_mapper["dbfs_path"])

    feedback = gen(dbfs_path, insurer, conf_mapper, 1)
    if mapper_lead:
        lead_map = lead_mapper_transform(spark, lead_mapper)
        feedback = feedback.join(lead_map, F.col(
            "src_file") == F.col("Filename"), how='left')
    
    return feedback

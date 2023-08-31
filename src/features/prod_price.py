from src.utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce

from .prod_hierarcy_recency import get_agg_total_store

@logger
def get_agg_prod_price(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F
    
    txn = txn.where(F.col("household_id") != -1)

    price_level_df = txn.groupBy('household_id','price_level')\
                        .agg(F.sum('net_spend_amt').alias('Spend'), \
                        F.count_distinct('transaction_uid').alias('Visits'), \
                            F.sum('unit').alias('Units'))
    total_df = get_agg_total_store(spark, conf_mapper, txn)
                                            
    price_level_df = price_level_df.join(total_df.where(F.col("household_id") != -1), on='household_id', how='inner')

    price_level_df = price_level_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))\
                .withColumn('PCT_PCT_Spend',F.col('Spend') /F.col('Total_Spend'))\
                .withColumn('PCT_PCT_Visits',F.col('Visits') /F.col('Total_Visits'))\
                .withColumn('PCT_PCT_Units',F.col('Units') /F.col('Total_Units'))
                

    pivot_columns = price_level_df.select("price_level").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_price_level_df = price_level_df.groupBy("household_id").pivot("price_level", pivot_columns).agg(
        F.first("Spend").alias("Spend"),
        F.first("Visits").alias("Visits"),
        F.first("Units").alias("Units"),
        F.first("SPV").alias("SPV"),
        F.first("UPV").alias("UPV"),
        F.first("SPU").alias("SPU"),
        F.first("PCT_Spend").alias("PCT_Spend"),
        F.first("PCT_Visits").alias("PCT_Visits"),
        F.first("PCT_Units").alias("PCT_Units"),
        F.first("PCT_PCT_Spend").alias("PCT_PCT_Spend"),
        F.first("PCT_PCT_Visits").alias("PCT_PCT_Visits"),
        F.first("PCT_PCT_Units").alias("PCT_PCT_Units")
    ).fillna(0)

    for c in pivot_columns:
        pivoted_price_level_df = pivoted_price_level_df.withColumnRenamed(c +"_PCT_Spend", "PCT_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_" + c + "_UNITS")\
                                    .withColumnRenamed(c +"_PCT_PCT_Spend", "PCT_PCT_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_PCT_Visits", "PCT_PCT_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_PCT_Units", "PCT_PCT_" + c + "_UNITS")

    return pivoted_price_level_df
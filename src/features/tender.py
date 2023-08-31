from src.utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce

from .prod_hierarcy_recency import get_agg_total_store

@logger
def get_agg_tender(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F

    payment_df = txn.groupBy('household_id','payment_flag')\
                        .agg(F.sum('net_spend_amt').alias('Spend'), \
                        F.count_distinct('transaction_uid').alias('Visits'), \
                            F.sum('unit').alias('Units'))

    total_df = get_agg_total_store(spark, conf_mapper, txn)
                                                
    payment_df = payment_df.join(total_df, on='household_id', how='inner')

    payment_df = payment_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))


    pivot_columns = payment_df.select("payment_flag").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_payment_df = payment_df.groupBy("household_id").pivot("payment_flag", pivot_columns).agg(
        F.first("Spend").alias("Spend"),
        F.first("Visits").alias("Visits"),
        F.first("Units").alias("Units"),
        F.first("SPV").alias("SPV"),
        F.first("UPV").alias("UPV"),
        F.first("SPU").alias("SPU"),
        F.first("PCT_Spend").alias("PCT_Spend"),
        F.first("PCT_Visits").alias("PCT_Visits"),
        F.first("PCT_Units").alias("PCT_Units")
    ).fillna(0)

    for c in pivot_columns:
        if (c == 'VOUCHER' or c == 'COUPON'):
            pivoted_payment_df = pivoted_payment_df.withColumnRenamed(c +"_Spend", "PYMNT_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "PYMNT_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "PYMNT_" + c + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "PYMNT_" + c + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "PYMNT_" + c + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "PYMNT_" + c + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_PYMNT_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_PYMNT_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_PYMNT_" + c + "_UNITS")
        else:
            pivoted_payment_df = pivoted_payment_df.withColumnRenamed(c +"_Spend", "PYMT_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "PYMT_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "PYMT_" + c + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "PYMT_" + c + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "PYMT_" + c + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "PYMT_" + c + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_PYMT_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_PYMT_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_PYMT_" + c + "_UNITS")
    return pivoted_payment_df
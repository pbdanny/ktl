from src.utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce
from .prod_hierarcy_recency import get_agg_total_store

@logger
def get_agg_store_format(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F
    
    total_df = get_agg_total_store(spark, conf_mapper, txn)
    
    txn_add_fmt = txn.replace({"HDE":"Hypermarket", "Talad":"Supermarket", "GoFresh":"MiniSupermarket"}, subset=["store_format_online_subchannel_other"])
    
    store_format_df = txn_add_fmt\
                        .where(F.col("store_format_online_subchannel_other").isNotNull())\
                        .groupBy('household_id','store_format_online_subchannel_other')\
                        .agg(F.sum('net_spend_amt').alias('Spend'), \
                        F.count_distinct('transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))
                        
    store_format_df = store_format_df.join(total_df, on='household_id', how='inner')

    store_format_df = store_format_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

    pivot_columns = store_format_df\
                    .where(F.col("store_format_online_subchannel_other").isNotNull())\
                    .select("store_format_online_subchannel_other").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_store_format_df = store_format_df.groupBy("household_id").pivot("store_format_online_subchannel_other", pivot_columns).agg(
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
        pivoted_store_format_df = pivoted_store_format_df.withColumnRenamed(c +"_Spend", "SF_" + c.upper().replace(" ","") + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "SF_" + c.upper().replace(" ","") + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "SF_" + c.upper().replace(" ","") + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "SF_" + c.upper().replace(" ","") + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "SF_" + c.upper().replace(" ","") + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "SF_" + c.upper().replace(" ","") + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_SF_" + c.upper().replace(" ","") + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_SF_" + c.upper().replace(" ","") + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_SF_" + c.upper().replace(" ","") + "_UNITS")
                                    
    return pivoted_store_format_df

@logger
def get_agg_store_region(spark, conf_mapper, txn):
    """
    """
    store_region_df = txn\
                        .where(F.col("store_region").isNotNull())\
                        .groupBy('household_id','store_region')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))

    total_df = get_agg_total_store(spark, conf_mapper, txn)
                                               
    store_region_df = store_region_df.join(total_df, on='household_id', how='inner')

    store_region_df = store_region_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

    pivot_columns = store_region_df.where(F.col("store_region").isNotNull())\
                                 .select("store_region").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_store_region_df = store_region_df.groupBy("household_id").pivot("store_region", pivot_columns).agg(
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
        pivoted_store_region_df = pivoted_store_region_df.withColumnRenamed(c +"_Spend", "SA_" + c.upper().replace(" ","") + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "SA_" + c.upper().replace(" ","") + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "SA_" + c.upper().replace(" ","") + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "SA_" + c.upper().replace(" ","") + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "SA_" + c.upper().replace(" ","") + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "SA_" + c.upper().replace(" ","") + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_SA_" + c.upper().replace(" ","") + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_SA_" + c.upper().replace(" ","") + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_SA_" + c.upper().replace(" ","") + "_UNITS")

                                
    return pivoted_store_region_df
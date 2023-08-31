from src.utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce

from .prod_hierarcy_recency import get_agg_total_store

@logger
def get_agg_promo(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F

    discount_promo_df = txn.groupBy('household_id','promo_flag')\
                        .agg(F.sum('net_spend_amt').alias('Spend'), \
                        F.count_distinct('transaction_uid').alias('Visits'), \
                            F.sum('unit').alias('Units'))

    total_df = get_agg_total_store(spark, conf_mapper, txn)

    discount_promo_df = discount_promo_df.join(total_df, on='household_id', how='inner')

    discount_promo_df = discount_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))


    pivot_columns = discount_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_discount_promo_df = discount_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
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
        if (c == 'PROMO' or c == 'MARK_DOWN'):
            pivoted_discount_promo_df = pivoted_discount_promo_df.withColumnRenamed(c +"_Spend", c + "_SPEND")\
                                        .withColumnRenamed(c +"_Visits", c + "_VISITS")\
                                        .withColumnRenamed(c +"_Units", c + "_UNITS")\
                                        .withColumnRenamed(c +"_PCT_Spend", "PCT_" + c + "_SPEND")\
                                        .withColumnRenamed(c +"_PCT_Visits", "PCT_" + c + "_VISITS")\
                                        .withColumnRenamed(c +"_PCT_Units", "PCT_" + c + "_UNITS")
        else:
            pivoted_discount_promo_df = pivoted_discount_promo_df.withColumnRenamed(c +"_Spend", "DISCOUNT_" + c + "_SPEND")\
                                        .withColumnRenamed(c +"_Visits", "DISCOUNT_" + c + "_VISITS")\
                                        .withColumnRenamed(c +"_Units", "DISCOUNT_" + c + "_UNITS")\
                                        .withColumnRenamed(c +"_SPV", "DISCOUNT_" + c + "_SPV")\
                                        .withColumnRenamed(c +"_UPV", "DISCOUNT_" + c + "_UPV")\
                                        .withColumnRenamed(c +"_SPU", "DISCOUNT_" + c + "_SPU")\
                                        .withColumnRenamed(c +"_PCT_Spend", "PCT_DISCOUNT_" + c + "_SPEND")\
                                        .withColumnRenamed(c +"_PCT_Visits", "PCT_DISCOUNT_" + c + "_VISITS")\
                                        .withColumnRenamed(c +"_PCT_Units", "PCT_DISCOUNT_" + c + "_UNITS")

    return pivoted_discount_promo_df

@logger
def get_agg_promo_time_of_day(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F

    time_promo_df = txn.groupBy('household_id','time_of_day','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))

    total_df = get_agg_total_store(spark, conf_mapper, txn)

    time_promo_df = time_promo_df.join(total_df, on='household_id', how='inner')

    time_promo_df = time_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))\
                .withColumn('time_promo',F.concat_ws('_',F.col('time_of_day'),F.col('promo_flag')))

    # time_promo_df.display()

    pivot_columns = time_promo_df.select("time_promo").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_time_promo_df = time_promo_df.groupBy("household_id").pivot("time_promo", pivot_columns).agg(
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
        pivoted_time_promo_df = pivoted_time_promo_df.withColumnRenamed(c +"_Spend", "TIME_" + c.upper() + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "TIME_" + c.upper() + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "TIME_" + c.upper() + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "TIME_" + c.upper() + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "TIME_" + c.upper() + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "TIME_" + c.upper() + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_TIME_" + c.upper() + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_TIME_" + c.upper() + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_TIME_" + c.upper() + "_UNITS")

    return pivoted_time_promo_df

@logger
def get_agg_promo_last_wkend(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F

    total_df = get_agg_total_store(spark, conf_mapper, txn)

    last_weekend_promo_df = txn.filter(F.col('last_weekend_flag') == 'Y')\
                       .groupBy('household_id','promo_flag')\
                       .agg(F.sum('net_spend_amt').alias('Spend'), \
                       F.count_distinct('transaction_uid').alias('Visits'), \
                        F.sum('unit').alias('Units'))

    last_weekend_promo_df = last_weekend_promo_df.join(total_df, on='household_id', how='inner')

    last_weekend_promo_df = last_weekend_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

    pivot_columns = last_weekend_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_last_weekend_promo_df = last_weekend_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
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
        pivoted_last_weekend_promo_df = pivoted_last_weekend_promo_df.withColumnRenamed(c +"_Spend", "LAST_WKND_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "LAST_WKND_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "LAST_WKND_" + c + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "LAST_WKND_" + c + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "LAST_WKND_" + c + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "LAST_WKND_" + c + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_LAST_WKND_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_LAST_WKND_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_LAST_WKND_" + c + "_UNITS")

    return pivoted_last_weekend_promo_df

@logger
def get_agg_promo_recency(spark, conf_mapper, txn, recency_col_nm:str):
    """
    """
    from pyspark.sql import functions as F
    MAPPER_RECENCY_NM_FEATURE_NM = {"":"",
                                    "last_3_flag": "L3",
                                    "last_6_flag": "L6",
                                    "last_9_flag": "L9"}
    features_rcncy_col_nm = MAPPER_RECENCY_NM_FEATURE_NM[recency_col_nm]

    l3_promo_df = txn.filter(F.col(recency_col_nm) == 'Y')\
                    .groupBy('household_id','promo_flag')\
                    .agg(F.sum('net_spend_amt').alias(f'{features_rcncy_col_nm}_SPEND'), \
                    F.count_distinct('transaction_uid').alias(f'{features_rcncy_col_nm}_VISITS'), \
                    F.sum('unit').alias(f'{features_rcncy_col_nm}_UNITS'))

    total_df = get_agg_total_store(spark, conf_mapper, txn)

    l3_promo_df = l3_promo_df.join(total_df, on='household_id', how='inner')

    l3_promo_df = l3_promo_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Visits')))\
               .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                 .otherwise(F.col('Units') /F.col('Visits')))\
               .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                 .otherwise(F.col('Spend') /F.col('Units')))\
               .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
               .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
               .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))

    pivot_columns = l3_promo_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_l3_promo_df = l3_promo_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
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
        pivoted_l3_promo_df = pivoted_l3_promo_df.withColumnRenamed(f"{features_rcncy_col_nm}"+"_Spend", "L3_" + f"{features_rcncy_col_nm}"+ "_SPEND")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_Visits", "L3_" + f"{features_rcncy_col_nm}"+ "_VISITS")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_Units", "L3_" + f"{features_rcncy_col_nm}"+ "_UNITS")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_SPV", "L3_" + f"{features_rcncy_col_nm}"+ "_SPV")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_UPV", "L3_" + f"{features_rcncy_col_nm}"+ "_UPV")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_SPU", "L3_" + f"{features_rcncy_col_nm}"+ "_SPU")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_PCT_Spend", "PCT_L3_" + f"{features_rcncy_col_nm}"+ "_SPEND")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_PCT_Visits", "PCT_L3_" + f"{features_rcncy_col_nm}"+ "_VISITS")\
                                    .withColumnRenamed(f"{features_rcncy_col_nm}"+"_PCT_Units", "PCT_L3_" + f"{features_rcncy_col_nm}"+ "_UNITS")
    return pivoted_l3_promo_df

@logger
def get_agg_promo_item_recency(spark, conf_mapper, txn, recency_col_nm:str):
    """
    """
    from pyspark.sql import functions as F

    MAPPER_RECENCY_NM_FEATURE_NM = {"":"",
                                    "last_3_flag": "L3",
                                    "last_6_flag": "L6",
                                    "last_9_flag": "L9"}

    features_rcncy_col_nm = MAPPER_RECENCY_NM_FEATURE_NM[recency_col_nm]

    l3_promo_item_df = txn.filter(F.col(recency_col_nm) == 'Y')\
                    .groupBy('household_id','promo_flag')\
                    .agg(F.sum('unit').alias('Items'))

    pivot_columns = l3_promo_item_df.select("promo_flag").distinct().rdd.flatMap(lambda x: x).collect()

    pivoted_l3_promo_item_df = l3_promo_item_df.groupBy("household_id").pivot("promo_flag", pivot_columns).agg(
    F.first("Items")
    ).fillna(0)

    for c in pivot_columns:
        pivoted_l3_promo_item_df = pivoted_l3_promo_item_df.withColumnRenamed(c, f"{features_rcncy_col_nm}_" + c + "_ITEMS")

    return pivoted_l3_promo_item_df
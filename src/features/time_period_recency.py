from src.utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce
from .prod_hierarcy_recency import get_agg_total_store

@logger
def get_agg_monthly(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F
    
    monthly_df = txn.groupBy('household_id', 'end_month_date')\
                                .agg(F.sum('net_spend_amt').alias('Spend'), \
                                F.countDistinct('transaction_uid').alias('Visits'), \
                                F.sum('unit').alias('Units'))\
                                .fillna(0)

    monthly_df = monthly_df.groupBy('household_id').agg(
            F.round(F.avg(F.coalesce(F.col("Spend"), F.lit(0))), 2).alias("AVG_SPEND_MNTH"),
            F.round(F.stddev(F.coalesce(F.col("Spend"), F.lit(0))), 2).alias("SD_SPEND_MNTH"),
            F.round(F.min(F.coalesce(F.col("Spend"), F.lit(0))), 2).alias("MIN_SPEND_MNTH"),
            F.round(F.max(F.coalesce(F.col("Spend"), F.lit(0))), 2).alias("MAX_SPEND_MNTH"),
            F.round(F.avg(F.coalesce(F.col("Visits"), F.lit(0))), 2).alias("AVG_VISITS_MNTH"),
            F.round(F.stddev(F.coalesce(F.col("Visits"), F.lit(0))), 2).alias("SD_VISITS_MNTH"),
            F.round(F.min(F.coalesce(F.col("Visits"), F.lit(0))), 2).alias("MIN_VISITS_MNTH"),
            F.round(F.max(F.coalesce(F.col("Visits"), F.lit(0))), 2).alias("MAX_VISITS_MNTH"),
            F.round(F.avg(F.coalesce(F.col("Units"), F.lit(0))), 2).alias("AVG_UNITS_MNTH"),
            F.round(F.stddev(F.coalesce(F.col("Units"), F.lit(0))), 2).alias("SD_UNITS_MNTH"),
            F.round(F.min(F.coalesce(F.col("Units"), F.lit(0))), 2).alias("MIN_UNITS_MNTH"),
            F.round(F.max(F.coalesce(F.col("Units"), F.lit(0))), 2).alias("MAX_UNITS_MNTH")
    ).fillna(0)
    
    return monthly_df

@logger
def get_agg_wkly(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F
    
    weekly_df = txn.groupBy('household_id', 'week_of_month')\
                                    .agg(F.sum('net_spend_amt').alias('Spend'), \
                                    F.countDistinct('transaction_uid').alias('Visits'), \
                                    F.sum('unit').alias('Units'))\
                                    .fillna(0)

    weekly_df = weekly_df.groupBy('household_id').agg(
            F.round(F.avg(F.coalesce(F.col("Spend"), F.lit(0))), 2).alias("AVG_SPEND_WK"),
            F.round(F.stddev(F.coalesce(F.col("Spend"), F.lit(0))), 2).alias("SD_SPEND_WK"),
            F.round(F.avg(F.coalesce(F.col("Visits"), F.lit(0))), 2).alias("AVG_VISITS_WK"),
            F.round(F.stddev(F.coalesce(F.col("Visits"), F.lit(0))), 2).alias("SD_VISITS_WK"),
            F.round(F.avg(F.coalesce(F.col("Units"), F.lit(0))), 2).alias("AVG_UNITS_WK"),
            F.round(F.stddev(F.coalesce(F.col("Units"), F.lit(0))), 2).alias("SD_UNITS_WK")
        ).fillna(0)
    
    return weekly_df

@logger
def get_agg_quarter(spark, conf_mapper, txn, quarter_col_nm:str):
    """
    """
    from pyspark.sql import functions as F
    
    total_df = get_agg_total_store(spark, conf_mapper, txn)
    
    MAPPER_COL_NM_FEATURE_NM = {"q1_flag": "Q1",
                                "q2_flag": "Q2",
                                "q3_flag": "Q3",
                                "q4_flag": "Q4",
                                }
    features_qrt_col_nm = MAPPER_COL_NM_FEATURE_NM[quarter_col_nm]
    
    print(f"Aggreate by time period at {quarter_col_nm}")
    
    qtr1_df = txn.filter(F.col(quarter_col_nm) == 'Y')\
                        .groupBy('household_id','app_year_qtr')\
                        .agg(sum('net_spend_amt').alias(f'{features_qrt_col_nm}_SPEND'), \
                            F.countDistinct('transaction_uid').alias(f'{features_qrt_col_nm}_VISITS'), \
                            F.sum('unit').alias(f'{features_qrt_col_nm}_UNITS'))
                                                
    qtr1_df = qtr1_df.join(total_df, on='household_id', how='inner')

    qtr1_df = qtr1_df.withColumn(f'{features_qrt_col_nm}_SPV', 
                                 F.when((F.col(f'{features_qrt_col_nm}_VISITS').isNull()) | (F.col(f'{features_qrt_col_nm}_VISITS') == 0), 0)\
                                    .otherwise(F.col(f'{features_qrt_col_nm}_SPEND') / F.col(f'{features_qrt_col_nm}_VISITS')))\
                .withColumn(f'{features_qrt_col_nm}_UPV', F.when((F.col(f'{features_qrt_col_nm}_VISITS').isNull()) | (F.col(f'{features_qrt_col_nm}_VISITS') == 0), 0)\
                                    .otherwise(F.col(f'{features_qrt_col_nm}_UNITS') / F.col(f'{features_qrt_col_nm}_VISITS')))\
                .withColumn(f'{features_qrt_col_nm}_SPU', F.when((F.col(f'{features_qrt_col_nm}_UNITS').isNull()) | (F.col(f'{features_qrt_col_nm}_UNITS') == 0), 0)\
                                    .otherwise(F.col(f'{features_qrt_col_nm}_SPEND') / F.col(f'{features_qrt_col_nm}_UNITS')))\
                .withColumn(f'PCT_{features_qrt_col_nm}_SPEND', F.col(f'{features_qrt_col_nm}_SPEND') * 100 / F.col('Total_Spend'))\
                .withColumn(f'PCT_{features_qrt_col_nm}_VISITS', F.col(f'{features_qrt_col_nm}_VISITS') * 100 / F.col('Total_Visits'))\
                .withColumn(f'PCT_{features_qrt_col_nm}_UNITS', F.col(f'{features_qrt_col_nm}_UNITS') * 100 / F.col('Total_Units'))\
                .drop('Total_Spend', 'Total_Visits', 'Total_Units', 'app_year_qtr')
                
    return qtr1_df

@logger
def get_agg_festive(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F
    
    fest_df = txn.groupBy('household_id', 'fest_flag')\
                                    .agg(F.sum('net_spend_amt').alias('Spend'), \
                                    F.countDistinct('transaction_uid').alias('Visits'), \
                                    F.sum('unit').alias('Units'))\
                                    .fillna(0)
                                    
    total_df = get_agg_total_store(spark, conf_mapper, txn)
    
    fest_df = fest_df.join(total_df, on='household_id', how='inner')

    fest_df = fest_df.withColumn('SPV', F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') / F.col('Visits')))\
                .withColumn('UPV', F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') / F.col('Visits')))\
                .withColumn('SPU', F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') / F.col('Units')))\
                .withColumn('PCT_Spend', F.col('Spend') * 100 / F.col('Total_Spend'))\
                .withColumn('PCT_Visits', F.col('Visits') * 100 / F.col('Total_Visits'))\
                .withColumn('PCT_Units', F.col('Units') * 100 / F.col('Total_Units'))

    pivot_columns = fest_df.select("fest_flag").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_fest_df = fest_df.groupBy("household_id").pivot("fest_flag", pivot_columns).agg(
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
        
        pivoted_fest_df = pivoted_fest_df.withColumnRenamed(c +"_Spend", "FEST_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "FEST_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "FEST_" + c + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "FEST_" + c + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "FEST_" + c + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "FEST_" + c + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_FEST_" + c + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_FEST_" + c + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_FEST_" + c + "_UNITS")
    return pivoted_fest_df

@logger
def get_agg_time_of_day(spark, conf_mapper, txn):
    """
    """
    from pyspark.sql import functions as F
    
    time_of_day = txn.groupBy('household_id', 'time_of_day')\
                                .agg(F.sum('net_spend_amt').alias('Spend'), \
                                F.countDistinct('transaction_uid').alias('Visits'), \
                                F.sum('unit').alias('Units'))\
                                .fillna(0)
                                
    total_df = get_agg_total_store(spark, conf_mapper, txn)
                                
    time_day_df = time_day_df.join(total_df, on='household_id', how='inner')
    
    time_day_df = time_day_df.withColumn('SPV', F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                .otherwise(F.col('Spend') / F.col('Visits')))\
            .withColumn('UPV', F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                .otherwise(F.col('Units') / F.col('Visits')))\
            .withColumn('SPU', F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                .otherwise(F.col('Spend') / F.col('Units')))\
            .withColumn('PCT_Spend', F.col('Spend') * 100 / F.col('Total_Spend'))\
            .withColumn('PCT_Visits', F.col('Visits') * 100 / F.col('Total_Visits'))\
            .withColumn('PCT_Units', F.col('Units') * 100 / F.col('Total_Units'))\
            .withColumn('PCT_PCT_Spend', F.col('PCT_Spend') / F.col('Total_Spend'))\
            .withColumn('PCT_PCT_Visits', F.col('PCT_Visits') / F.col('Total_Visits'))\
            .withColumn('PCT_PCT_Units', F.col('PCT_Units') / F.col('Total_Units'))
            
    pivot_columns = time_day_df.select("fest_flag").distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_time_day_df = time_day_df.groupBy("household_id").pivot("time_of_day", pivot_columns).agg(
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
        
        pivoted_time_day_df = pivoted_time_day_df.withColumnRenamed(c +"_Spend", "FEST_" + c.upper() + "_SPEND")\
                                    .withColumnRenamed(c +"_Visits", "FEST_" + c.upper() + "_VISITS")\
                                    .withColumnRenamed(c +"_Units", "FEST_" + c.upper() + "_UNITS")\
                                    .withColumnRenamed(c +"_SPV", "FEST_" + c.upper() + "_SPV")\
                                    .withColumnRenamed(c +"_UPV", "FEST_" + c.upper() + "_UPV")\
                                    .withColumnRenamed(c +"_SPU", "FEST_" + c.upper() + "_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", "PCT_FEST_" + c.upper() + "_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", "PCT_FEST_" + c.upper() + "_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", "PCT_FEST_" + c.upper() + "_UNITS")

    return pivoted_time_day_df


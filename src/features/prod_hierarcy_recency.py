from src.utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce

@logger
def get_agg_total_store(spark, conf_mapper, txn):
    """
    """
    agg_total = txn.groupBy('household_id')\
                   .agg(F.sum('net_spend_amt').alias('Total_Spend'), \
                        F.count_distinct('transaction_uid').alias('Total_Visits'), \
                        F.sum('unit').alias('Total_Units'))
                   
    return agg_total

@logger
def get_agg_prd_hier_recency(spark, conf_mapper, txn, prod_hier_id_col_nm:str, recency_col_nm:str):
    """
    """
    from pyspark.sql import functions as F
    
    total_df = get_agg_total_store(spark, conf_mapper, txn)
    
    MAPPER_COL_NM_FEATURE_NM = {"division_id": "DIV",
                                "grouped_department_code": "DEP",
                                "grouped_section_code": "SEC"
                                }
    features_prod_col_nm = MAPPER_COL_NM_FEATURE_NM[prod_hier_id_col_nm]
    
    dep_exclude = ['1_36','1_92','13_25','13_32']
    sec_exclude = ['3_7_130', '3_7_131', '3_8_132', '3_9_81', '10_43_34', '3_14_78', '13_6_205', '13_67_364',
        '1_36_708', '1_45_550', '1_92_992', '2_3_245', '2_4_253', '2_66_350', '13_25_249', '2_4_253',
        '13_25_250', '13_25_251', '13_67_359', '2_66_350', '4_10_84', '4_56_111', '10_46_549',
        '13_6_316', '13_25_249', '13_25_250', '13_25_251', '13_67_359', '13_32_617', '13_67_360', '2_4_719']
    
    MAPPER_COL_NM_EXCLUSION = {"division_id": "",
                                "grouped_department_code": dep_exclude,
                                "grouped_section_code": sec_exclude
                                } 
    
    exclusion_list = MAPPER_COL_NM_EXCLUSION[prod_hier_id_col_nm]
    print(f"Aggregate KPI for by {prod_hier_id_col_nm} with recency {recency_col_nm} and exclusion {exclusion_list}")
    
    div_df = txn.filter((F.col(prod_hier_id_col_nm).isNotNull()))\
                .filter(~(F.col(prod_hier_id_col_nm).isin(exclusion_list)))\
                    .groupBy('household_id', prod_hier_id_col_nm)\
                        .agg(F.sum('net_spend_amt').alias('Spend'), \
                        F.count_distinct('transaction_uid').alias('Visits'), \
                            F.sum('unit').alias('Units'))
                        
    div_df = div_df.join(total_df, on='household_id', how='inner')

    div_df = div_df.withColumn('SPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Visits')))\
                .withColumn('UPV',F.when((F.col('Visits').isNull()) | (F.col('Visits') == 0), 0)\
                                    .otherwise(F.col('Units') /F.col('Visits')))\
                .withColumn('SPU',F.when((F.col('Units').isNull()) | (F.col('Units') == 0), 0)\
                                    .otherwise(F.col('Spend') /F.col('Units')))\
                .withColumn('PCT_Spend',F.col('Spend') * 100 /F.col('Total_Spend'))\
                .withColumn('PCT_Visits',F.col('Visits') * 100 /F.col('Total_Visits'))\
                .withColumn('PCT_Units',F.col('Units') * 100 /F.col('Total_Units'))
                
    pivot_columns = div_df.select(prod_hier_id_col_nm).distinct().rdd.flatMap(lambda x: x).collect()
    pivoted_div_df = div_df.groupBy("household_id").pivot(prod_hier_id_col_nm, pivot_columns).agg(
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
    
    MAPPER_RECENCY_NM_FEATURE_NM = {"":"",
                                    "last_3_flag": "_L3",
                                    "last_6_flag": "_L6",
                                    "last_9_flag": "_L9"}

    features_rcncy_col_nm = MAPPER_RECENCY_NM_FEATURE_NM[recency_col_nm]

    for c in pivot_columns:
        c = str(c)
        pivoted_div_df = pivoted_div_df.withColumnRenamed(c +"_Spend", f"CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_SPEND")\
                                    .withColumnRenamed(c +"_Visits", f"CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_VISITS")\
                                    .withColumnRenamed(c +"_Units", f"CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_UNITS")\
                                    .withColumnRenamed(c +"_SPV", f"CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_SPV")\
                                    .withColumnRenamed(c +"_UPV", f"CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_UPV")\
                                    .withColumnRenamed(c +"_SPU", f"CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_SPU")\
                                    .withColumnRenamed(c +"_PCT_Spend", f"PCT_CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_SPEND")\
                                    .withColumnRenamed(c +"_PCT_Visits", f"PCT_CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_VISITS")\
                                    .withColumnRenamed(c +"_PCT_Units", f"PCT_CAT_{features_prod_col_nm}_%" + c + f"%{features_rcncy_col_nm}_UNITS")

    #exclude the dummy customer
    pivoted_div_df = pivoted_div_df.filter(~(F.col('household_id') == -1))

    return pivoted_div_df
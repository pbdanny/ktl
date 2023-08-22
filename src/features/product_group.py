from utils.logger import logger
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
import sys
import os
from functools import reduce

sys.path.append('../')


@logger
def load_prod_group(spark):
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    conf_mapper = files.conf_reader("../config/transaction.json")
    # use prod_gr_prefix instead of abfss_prefix
    abfss_prefix = mnt_mapper["prod_gr_prefix"]
    prd_gr_ver = conf_mapper["prd_gr_ver"]
    product_grp = spark.read.parquet(os.path.join(
        abfss_prefix, "product_group", f"prod_grp_{prd_gr_ver}.parquet"))

    product_grp.select("flag_product_group_BKH_UC").drop_duplicates().display()

    prd_gr_flag = \
        (product_grp
         .select("upc_id", F.lower("flag_product_group_BKH_UC").alias("prod_gr"))
         .withColumn("baby", F.when(F.col("prod_gr").isin(["baby", "baby_kids", "babyaccessories", "babysupplement"]), "Yes").otherwise("No"))
         .withColumn("child", F.when(F.col("prod_gr").isin(["kid"]), "Yes").otherwise("No"))
         .withColumn("healthy", F.when(F.col("prod_gr").isin(["health_prod"]), "Yes").otherwise("No"))
         )
    return prd_gr_flag


@logger
def get_cust_prod(txn_cust_r360, prd_gr_flag):
    from .recency import kpi
    cust_total = \
        (txn_cust_r360
         .groupBy("household_id")
         .agg(*kpi.kpi("total", None))
         )
    cust_prod_gr = \
        (txn_cust_r360
         .join(prd_gr_flag, on="upc_id", how="left")
         .fillna(value="NA", subset=['prod_gr'])
         .groupBy("household_id", "prod_gr")
         .agg(*kpi.kpi(None, None))
         )

    cust_prop_prod_gr = \
        (cust_total
         .join(cust_prod_gr, on="household_id", how="left")
         .fillna(0)
         .withColumn("prop_sales", F.col("sales")/F.col("total_sales"))
         .withColumn("prop_units", F.col("units")/F.col("total_units"))
         .withColumn("prop_visits", F.col("visits")/F.col("total_visits"))
         .select("household_id", "prod_gr", "prop_sales", "prop_units", "prop_visits")
         )
    return cust_total, cust_prop_prod_gr


@logger
def cust_visit_flag(txn_cust_r360, prd_gr_flag, condition_filter, col_name):
    '''
        function to get customer visit flag
        :param:
            txn_cust_r360: spark dataframe, txn_cust_r360
            prd_gr_flag: spark dataframe, product group flag
            condition_filter: string, condition to filter the product group
            col_name: string, column name for the flag
    '''
    cust_visit = \
        (txn_cust_r360
         .join(prd_gr_flag, on="upc_id", how="left")
         .where(F.col(condition_filter) == "Yes")
         .select("household_id")
         .drop_duplicates()
         .withColumn(col_name, F.lit("Yes"))
         )
    return cust_visit


@logger
def main(spark, prjct_nm, test):
    '''
        main() function to run the script
        :param: 
            spark spark session
            prjct_nm project name, e.g. edm_202210
            test: True/False, if True, only run on 1 week
    '''
    from . import get_txn_cust
    from .recency import kpi
    from utils import files
    mnt_mapper = files.conf_reader("../config/mnt.json")
    abfss_prefix = mnt_mapper["abfss_prefix"]
    txn_cust = get_txn_cust(spark, prjct_nm, test)
    txn_cust_r360 = txn_cust.where(F.col("date_id").between(
        F.col("r360_start_date"), F.col("first_contact_date")))
    prd_gr_flag = load_prod_group(spark)
    """
    (1) KPI by product group
    ----
    Recency : 360 days
    Store_Format : allFmt

    - prop sales by product group
    - prop units by product group
    - prop visits by product group

    * Use dummy customer with all product group
    To preserved number of column after pivot
    """
    cust_total, cust_prop_prod_gr = get_cust_prod(txn_cust_r360, prd_gr_flag)
    # KPI by product group
    # Create dummy customer, with all product group for pivot & perserved all features column
    dummy_cust_all_prod_gr = \
        (prd_gr_flag
         .select("prod_gr")
         .drop_duplicates()
         .withColumn('household_id', F.lit(9999))  # dummy cust
         .withColumn('prop_sales', F.lit(None))
         .withColumn('prop_units', F.lit(None))
         .withColumn('prop_visits', F.lit(None))
         )

    cust_prop_prod_gr_pv = \
        (cust_prop_prod_gr.unionByName(dummy_cust_all_prod_gr)
         .groupBy("household_id")
         .pivot("prod_gr")
         .agg(
            F.first("prop_sales").alias("prop_sales"),
            F.first("prop_units").alias("prop_units"),
            F.first("prop_visits").alias("prop_visits"),
        )
            .where(F.col("household_id") != 9999)  # remove dummy cust
            .fillna(0)
        )
    # check if all customer are in the table
    cust_prop_prod_gr_pv.agg(F.count_distinct("household_id")).show()

    filename = "cust_prop_prod_gr_pv.parquet" if not test else "cust_prop_prod_gr_pv_test.parquet"
    files.save(cust_prop_prod_gr_pv, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)
    """
    (2) Flag dominant Product group
    ----
    Dominant product group select from top % visits of those product group

    - Flag "Ever Child" use KIDS product group  
    - Flag "Ever Babay" use BABY, Baby_Kids, BABYACCESSORIES, BABYSUPPLEMENT  
    - Flag "Ever Healthy" use health_prod  
    """
    cust_dmnt_prod_gr = \
        (cust_prop_prod_gr
         .withColumn("row_nm_visit_sales",
                     F.row_number().over(Window.partitionBy("household_id").orderBy(F.col("prop_visits").desc(), F.col("prop_sales").desc_nulls_last())))
         .where(F.col("row_nm_visit_sales") == 1)
         .select("household_id", F.col("prod_gr").alias("dmnt_prod_gr"))
         )
    prod_groups = ["baby", "child", "healthy"]
    cust_visit = {}
    for col in prod_groups:
        cust_visit[col] = cust_visit_flag(
            txn_cust_r360, prd_gr_flag, col, "shop_"+col)
    cust_prod_flag = (cust_total.join(cust_dmnt_prod_gr, on="household_id", how="left").fillna(
        value="NA", subset=["dmnt_prod_gr"]))
    for col in prod_groups:
        cust_prod_flag = cust_prod_flag.join(
            cust_visit[col], on="household_id", how="left")
    cust_prod_flag = cust_prod_flag.fillna(
        value="No", subset=["shop_baby", "shop_child", "shop_healthy"])
    cust_prod_flag.show(10)
    filename = "cust_dmnnt_prod_gr_and_flag.parquet" if not test else "cust_dmnnt_prod_gr_and_flag_test.parquet"
    files.save(cust_prod_flag, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)

    """
    (3) % visit by shop mission
    ----
    - Shop mission id = Null : trader shop mission

    """

    sm_lkp = spark.table('tdm.srai_shopping_missions_lookup')

    cust_total_visit = \
        (txn_cust_r360
         .groupBy("household_id")
         .agg(*kpi.kpi("total", None, flag=["net_spend_amt", "transaction_uid"]))
         )

    cust_visit_by_sm = \
        (txn_cust_r360
         .join(sm_lkp, "shopping_missions_id", "left")
         .fillna("trader", subset=["shopping_missions"])
         .withColumn('shp_ms', F.regexp_replace(F.col("shopping_missions"), "\W+", ""))
         .groupBy("household_id", "shp_ms")
         .agg(*kpi.kpi(None, None, flag=["net_spend_amt", "transaction_uid"]))
         )

    cust_prop_sm_pv = \
        (cust_total_visit
         .join(cust_visit_by_sm, "household_id", "inner")
         .withColumn("prop_visits", F.col("visits")/F.col("total_visits"))
         .withColumn("prop_sales", F.col("sales")/F.col("total_sales"))
         .groupBy("household_id")
         .pivot("shp_ms")
         .agg(F.first("prop_visits").alias("prop_visits_360_days"),
              F.first("prop_sales").alias("prop_sales_360_days"),
              )
         .fillna(0)
         )

    cust_prop_sm_pv.show(10)
    filename = "cust_prop_sm_pv.parquet" if not test else "cust_prop_sm_pv_test.parquet"
    files.save(cust_prop_sm_pv, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)

    """
    (4) Ever shop mission flag
    ----
    - Flag only shop mission in 6 groups, excluding "Other" and null (trader shp mission)
    """

    cust_flag_sm_pv = \
        (txn_cust_r360
         # inner , remove null (trader)
         .join(sm_lkp, "shopping_missions_id", "inner")
         .where(~F.col("shopping_missions").isin(["Others"]))  # remove Ohers
         .withColumn('shp_ms', F.regexp_replace(F.col("shopping_missions"), "\W+", ""))
         .groupBy("household_id")
         .pivot("shp_ms")
         .agg(F.when(F.count("transaction_uid") > 0, F.lit("Yes")))
         .fillna("No")
         )

    cust_flag_sm_pv.show(10)
    filename = "cust_flag_sm_pv.parquet" if not test else "cust_flag_sm_pv_test.parquet"
    files.save(cust_flag_sm_pv, os.path.join(abfss_prefix, prjct_nm, "features",
               filename), format="parquet", mode="overwrite", overwriteSchema=True)

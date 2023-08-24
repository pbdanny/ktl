from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql import SparkSession

from src.utils.logger import logger
import os
import sys

sys.path.append('../')

@logger
def get_txn_cc_exc_trdr(spark, conf_mapper):
        
    start_week = conf_mapper["data"]["start_week"]
    end_week = conf_mapper["data"]["end_week"]
    timeframe_start = conf_mapper["data"]["timeframe_start"]
    timeframe_end = conf_mapper["data"]["timeframe_end"]

    txn_cc = (spark.table("tdm_seg.v_latest_txn118wk")
           .where(F.col("week_id").between(start_week, end_week))
           .where(F.col("date_id").between(timeframe_start, timeframe_end))
           .where(F.col("cc_flag").isin(["cc"]))
           .withColumn("store_region", F.when(F.col("store_region").isNull(), "Unidentified").otherwise(F.col("store_region")))
    )

    # Remove trader 
    trader_df = spark.table('tdm_seg.trader2023_subseg_master')
    max_quarter_id = 0
    max_quarter_id_trader = trader_df.agg(F.max('quarter_id')).collect()[0][0]
    max_txn_date_id = txn_cc.agg(F.max('date_id')).collect()[0][0]
    max_quarter_id_txn = spark.table("tdm.v_date_dim").where(F.col("date_id")==timeframe_end).select("quarter_id").collect()[0][0]

    if (max_quarter_id_trader >= max_quarter_id_txn):
        max_quarter_id = max_quarter_id_txn
    else:
        max_quarter_id = max_quarter_id_trader

    trader_df = spark.table('tdm_seg.trader2023_subseg_master').filter(F.col('quarter_id') == max_quarter_id)
    data_df = txn_cc.join(trader_df, on='household_id', how='leftanti')

    return data_df

@logger
def map_txn_time(spark, conf_mapper, txn):
    
    decision_date = conf_mapper["data"]["decision_date"]
    start_week = conf_mapper["data"]["start_week"]
    end_week = conf_mapper["data"]["end_week"]
    timeframe_start = conf_mapper["data"]["timeframe_start"]
    timeframe_end = conf_mapper["data"]["timeframe_end"]

    scope_date_dim = (spark
                .table('tdm.v_date_dim')
                .select(['date_id','period_id','quarter_id','year_id','month_id','weekday_nbr','week_id',
                        'day_in_month_nbr','day_in_year_nbr','day_num_sequence','week_num_sequence'])
                .where(F.col("week_id").between(start_week, end_week))
                .where(F.col("date_id").between(timeframe_start, timeframe_end))
                        .dropDuplicates()
                )

    time_of_day_df = (txn
                    .join(scope_date_dim.drop("week_id"), "date_id", "inner")
                    .withColumn('decision_date', F.lit(decision_date))
                    .withColumn('tran_hour', F.hour(F.col('tran_datetime')))
    )

    time_of_day_df = (time_of_day_df
                    .withColumn('time_of_day', F.when((F.col('tran_hour') >= 5) & (F.col('tran_hour') <= 8), 'prework')
                                                .when((F.col('tran_hour') >= 9) & (F.col('tran_hour') <= 11), 'morning')
                                                .when(F.col('tran_hour') == 12, 'lunch')
                                                .when((F.col('tran_hour') >= 13) & (F.col('tran_hour') <= 17), 'afternoon')
                                                .when((F.col('tran_hour') >= 18) & (F.col('tran_hour') <= 20), 'evening')
                                                .when(F.col('tran_hour') >= 21, 'late')
                                                .when(F.col('tran_hour') <= 4, 'night')
                                                .otherwise('def'))
                    .withColumn('week_of_month', 
                                F.when(F.col('day_in_month_nbr') <= 7, 1)
                                .when((F.col('day_in_month_nbr') > 7) & (F.col('day_in_month_nbr') <= 14), 2)
                                .when((F.col('day_in_month_nbr') > 14) & (F.col('day_in_month_nbr') <= 21), 3)
                                .when(F.col('day_in_month_nbr') > 21, 4))
                    .withColumn('weekend_flag', 
                                F.when(F.col('weekday_nbr').isin(6,7), F.lit('Y'))
                                .when((F.col('weekday_nbr') == 5) & (F.col('time_of_day').isin('evening', 'late')), 'Y')
                                .otherwise('N'))
    )    
    # festive week : Songkran, NewYear
    max_week_december = (scope_date_dim
                            .filter((F.col("month_id") % 100) == 12)
                            .filter(F.col("week_id").startswith(F.col("month_id").substr(1, 4))) 
                            .agg(F.max(F.col("week_id")).alias("max_week_december")).collect()[0]["max_week_december"]
        )

    d = scope_date_dim.select('week_id').distinct()

    df_with_lag_lead = d.withColumn("lag_week_id", F.lag("week_id").over(Window.orderBy("week_id"))) \
                        .withColumn("lead_week_id", F.lead("week_id").over(Window.orderBy("week_id")))

    week_before = df_with_lag_lead.filter(F.col("week_id") == max_week_december).select("lag_week_id").first()[0]
    week_after = df_with_lag_lead.filter(F.col("week_id") == max_week_december).select("lead_week_id").first()[0]

    xmas_week_id = [week_before, max_week_december, week_after]

    time_of_day_df = time_of_day_df.withColumn('fest_flag',F.when(F.col('week_id').isin(xmas_week_id), 'XMAS')\
                                                        .when(F.col('month_id').cast('string').endswith('04'), 'APRIL')\
                                                        .otherwise('NONE'))
    


    # Last week of month (Payweek)
    last_sat = scope_date_dim.filter(F.col('weekday_nbr') == 6).groupBy('month_id').agg(F.max('day_in_month_nbr').alias('day_in_month_nbr'))\
                                                    .withColumn('last_weekend_flag',F.lit('Y'))

    last_sat_df = scope_date_dim.select('date_id', 'month_id', 'day_in_month_nbr')\
                        .join(last_sat, on=['month_id','day_in_month_nbr'],how='inner')

    last_weekend_df = last_sat_df.select(F.col('month_id'),F.col('day_in_month_nbr'),F.col('date_id'),F.col('last_weekend_flag')) \
                    .unionAll(last_sat_df.select(F.col('month_id'),F.col('day_in_month_nbr'), F.date_add(F.col('date_id'), 1).alias('date_id'),F.col('last_weekend_flag'))) \
                    .unionAll(last_sat_df.select(F.col('month_id'),F.col('day_in_month_nbr'), F.date_sub(F.col('date_id'), 1).alias('date_id'),F.col('last_weekend_flag')))

    last_weekend_df = last_weekend_df.select('date_id', 'last_weekend_flag')

    flagged_df = (time_of_day_df
                .join(last_weekend_df, on='date_id',how='left')
                .fillna('N', subset=['last_weekend_flag'])
    )

    # Recency Flag
    r = flagged_df.withColumn('end_date',F.lit(timeframe_end))\
            .withColumn('start_date',F.lit(timeframe_start))\
            .withColumn('start_month_date', F.trunc(F.col('date_id'), 'month'))\
            .withColumn('end_month_date', F.last_day(F.col('start_month_date')))\
            .withColumn('months_from_end_date', F.months_between(F.col('end_date'),F.col('end_month_date')) + 1)\
            .withColumn('last_3_flag',F.when(F.col('months_from_end_date') <= 3 , 'Y')\
                                        .otherwise('N'))\
            .withColumn('last_6_flag',F.when(F.col('months_from_end_date') <= 6 , 'Y')\
                                                .otherwise('N'))\
            .withColumn('last_9_flag',F.when(F.col('months_from_end_date') <= 9 , 'Y')\
                                        .otherwise('N'))\
            .withColumn('q1_flag',F.when(F.col('months_from_end_date') <= 3 , 'Y')\
                                                .otherwise('N'))\
            .withColumn('q2_flag',F.when((F.col('months_from_end_date') > 3) & (F.col('months_from_end_date') <= 6) , 'Y')\
                                        .otherwise('N'))\
            .withColumn('q3_flag',F.when((F.col('months_from_end_date') > 6) & (F.col('months_from_end_date') <= 9) , 'Y')\
                                                .otherwise('N'))\
            .withColumn('q4_flag',F.when(F.col('months_from_end_date') > 9 , 'Y')\
                                                .otherwise('N'))\
            .withColumn('app_year_qtr',F.when(F.col('q1_flag') == 'Y', 'Q1')\
                                        .when(F.col('q2_flag') == 'Y', 'Q2')\
                                        .when(F.col('q3_flag') == 'Y', 'Q3')\
                                        .when(F.col('q4_flag') == 'Y', 'Q4')\
                                        .otherwise('NA'))

    return r

@logger
def map_txn_prod_premium(spark, conf_mapper, txn):
    """
    """
    # Premium / Budget Flag
    product_df = (spark.table('tdm.v_prod_dim_c')
                .select(['upc_id','brand_name','division_id','division_name','department_id','department_name','department_code','section_id','section_name','section_code','class_id','class_name','class_code','subclass_id','subclass_name','subclass_code'])
                .filter(F.col('division_id').isin([1,2,3,4,9,10,13]))
                .filter(F.col('country').isin("th"))
    )

    temp_prod_df = product_df.select('upc_id', 'subclass_code', 'subclass_name')

    premium_prod_df = (temp_prod_df
                    .filter(F.col('subclass_name').ilike('%PREMIUM%'))
                    .filter(~F.col('subclass_name').ilike('%COUPON%'))
                    .withColumn('price_level',F.lit('PREMIUM'))
                    ).distinct()

    budget_prod_df = (temp_prod_df
                    .filter(F.col('subclass_name').rlike('(?i)(budget|basic|value)'))
                    .withColumn('price_level',F.lit('BUDGET'))
                    ).distinct()

    price_level_df = premium_prod_df.unionByName(budget_prod_df)

    flagged_df = txn.join(price_level_df.select('upc_id','price_level'), on='upc_id', how='left')\
                    .fillna('NONE', subset=['price_level'])\
                    .dropna(subset=['household_id'])
    
    return flagged_df

@logger
def map_txn_tender(spark, conf_mapper, txn):
    """
    """

    timeframe_start = conf_mapper["data"]["timeframe_start"]
    timeframe_end = conf_mapper["data"]["timeframe_end"]
    
    # RESA tender
    resa_tender = spark.table("tdm.v_resa_group_resa_tran_tender")
    resa_tender = (
        resa_tender.withColumn("tender_type_group", F.trim(F.col("tender_type_group")))
        .withColumn(
            "set_tndr_type",
            F.array_distinct(
                F.collect_list(F.col("tender_type_group")).over(
                    Window.partitionBy(["tran_seq_no", "store", "day"])
                )
            ),
        )
        # .withColumn("set_tndr_type", F.collect_set(F.col("tender_type_group")).over(Window.partitionBy("tran_seq_no")))
        .withColumn("n_tndr_type", F.size(F.col("set_tndr_type")))
        .select(
            "tran_seq_no", "store", "day", "dp_data_dt", "n_tndr_type", "tender_type_group"
        )
        .withColumn(
            "sngl_tndr_type",
        F.when(F.col("n_tndr_type") == 1,F.col("tender_type_group")).otherwise(
            F.lit("MULTI")
            ),
        )
        # Adjust to support new unique txn_uid from surrogate key
        .withColumnRenamed("tran_seq_no", "transaction_uid_orig")
        .withColumnRenamed("store", "store_id")
        .withColumnRenamed("dp_data_dt", "date_id")
        .select("transaction_uid_orig", "store_id", "day", "date_id", "sngl_tndr_type")
        .drop_duplicates()
    )

    # OSM (Online) Tender
    oms_tender = spark.table("tdm_seg.v_oms_group_payment").filter(F.col("Country") == "th")

    oms_tender = (
        oms_tender.withColumn("PaymentMethod", F.trim(F.col("PaymentMethod")))
        .withColumn(
            "set_tndr_type",
            F.array_distinct(
                F.collect_list(F.col("PaymentMethod")).over(
                    Window.partitionBy(["transaction_uid"])
                )
            ),
        )
        # .withColumn("set_tndr_type", F.collect_set(F.col("tender_type_group")).over(Window.partitionBy("tran_seq_no")))
        .withColumn("n_tndr_type", F.size(F.col("set_tndr_type")))
        .select(
            "transaction_uid", "dp_data_dt", "n_tndr_type", "PaymentMethod"
        )
        .withColumn(
            "sngl_tndr_type",
        F.when(F.col("n_tndr_type") == 1,F.col("PaymentMethod")).otherwise(
            F.lit("MULTI")
            ),
        )
        # Adjust to support new unique txn_uid from surrogate key
        .withColumnRenamed("transaction_uid", "transaction_uid_orig")
        .select("transaction_uid_orig", "sngl_tndr_type")
        .drop_duplicates()
    )

    resa_tender = resa_tender.withColumn("oms",F.lit(False))
    oms_tender = oms_tender.withColumn("oms",F.lit(True))

    filter_resa_tender = resa_tender.filter(F.col('date_id').between(timeframe_start, timeframe_end))

    filter_resa_tender = filter_resa_tender.withColumnRenamed('transaction_uid_orig', 'transaction_uid')\
                                        .withColumnRenamed('sngl_tndr_type', 'resa_payment_method')\
                                        .dropDuplicates()

    filter_oms_tender = oms_tender.withColumnRenamed('sngl_tndr_type', 'oms_payment_method')\
                                .withColumnRenamed('transaction_uid_orig', 'transaction_uid')\
                                .dropDuplicates()

    # Add transaction type to txn
    flag_df = txn.join(filter_resa_tender.select('transaction_uid','store_id','date_id','resa_payment_method'), 
                                        on=['transaction_uid', 'store_id', 'date_id'], how='left')\
                                .join(filter_oms_tender.select('transaction_uid','oms_payment_method'), on='transaction_uid', how='left')

    flag_df = (flag_df
            .withColumn('resa_payment_method',
                        F.when(F.col('resa_payment_method').isNull(), F.lit('Unidentified'))
                            .otherwise(F.col('resa_payment_method')))
            .withColumn('payment_flag',
                        F.when((F.col('resa_payment_method') == 'CASH') | (F.col('oms_payment_method') == 'CASH'), 'CASH')
                            .when((F.col('resa_payment_method') == 'CCARD') | (F.col('oms_payment_method') == 'CreditCard'), 'CARD')
                                .when((F.col('resa_payment_method') == 'COUPON'), 'COUPON')
                                .when((F.col('resa_payment_method') == 'VOUCH'), 'VOUCHER')
                                .otherwise('OTHER'))
    )
    
    return flag_df

@logger
def map_txn_cust_issue_first_txn(spark, conf_mapper, txn):
    """
    """
    #--- get card_issue_date for nulls (use first transaction instead)

    max_card_issue = (spark
                    .table('tdm.v_customer_dim')
                    .groupBy("household_id", "golden_record_external_id_hash")
                    .agg(F.max("card_issue_date").alias("card_issue_date"))
                    .drop_duplicates()
    )

    first_tran = (spark.table('tdm_seg.mylotuss_customer_1st_txn_V1')
                .select('golden_record_external_id_hash', 'tran_datetime')
                .withColumnRenamed('tran_datetime', 'first_tran_datetime')
                )
                
    flag_df = (
        txn
        .join(max_card_issue, on='household_id', how='left')
        .join(first_tran, on='golden_record_external_id_hash', how='left')
        .withColumn('card_issue_date',
                    F.when(F.col('card_issue_date').isNull(),
                        F.col('first_tran_datetime')).otherwise(F.col('card_issue_date')))
        .withColumn('CC_TENURE', F.round((F.datediff(F.col('end_date'),F.col('card_issue_date'))) / 365,1))
        .withColumn('one_year_history',F.when(F.col('first_tran_datetime') <=F.col('date_id'), 1).otherwise(0))
        )
    return flag_df

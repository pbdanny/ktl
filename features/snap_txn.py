from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql import SparkSession

from utils.logger import logger
import os
import sys

sys.path.append('../')

@logger
def get_txn_cc_exc_trdr(spark, conf_path):
    
    from utils import files
    conf_mapper = files.conf_reader(conf_path)

    txn_cc = (spark.table("tdm_seg.v_latest_txn118wk")
           .where(F.col("week_id").between(conf_mapper["start_week"], conf_mapper["end_week"]))
           .where(F.col("date_id").between(conf_mapper["timeframe_start"], conf_mapper["timeframe_end"]))
           .where(F.col("cc_flag").isin(["cc"]))
           .withColumn("store_region", F.when(F.col("store_region").isNull(), "Unidentified").otherwise(F.col("store_region")))
    )

    # Remove trader 
    trader_df = spark.table('tdm_seg.trader2023_subseg_master')
    max_quarter_id = 0
    max_quarter_id_trader = trader_df.agg(F.max('quarter_id')).collect()[0][0]
    max_txn_date_id = txn_cc.agg(F.max('date_id')).collect()[0][0]
    max_quarter_id_txn = spark.table("tdm.v_date_dim").where(F.col("date_id")==conf_mapper["timeframe_end"]).select("quarter_id").collect()[0][0]

    if (max_quarter_id_trader >= max_quarter_id_txn):
        max_quarter_id = max_quarter_id_txn
    else:
        max_quarter_id = max_quarter_id_trader

    trader_df = spark.table('tdm_seg.trader2023_subseg_master').filter(F.col('quarter_id') == max_quarter_id)
    data_df = txn_cc.join(trader_df, on='household_id', how='leftanti')

    return data_df

def map_txn_time(spark, conf_path, txn):
    from utils import files
    conf_mapper = files.conf_reader(conf_path)

    date_dim = (spark
                .table('tdm.v_date_dim')
                .select(['date_id','period_id','quarter_id','year_id','month_id','weekday_nbr','week_id',
                        'day_in_month_nbr','day_in_year_nbr','day_num_sequence','week_num_sequence'])
                .where(F.col("week_id").between(conf_mapper["start_week"], conf_mapper["end_week"]))
                .where(F.col("date_id").between(conf_mapper["timeframe_start"], conf_mapper["timeframe_end"]))
                        .dropDuplicates()
                )

    time_of_day_df = (txn
                    .join(date_dim.drop("week_id"), "date_id", "inner")
                    .withColumn('decision_date', F.lit(conf_mapper["decision_date"]))
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
    max_week_december = (date_dim
                            .filter((F.col("month_id") % 100) == 12)
                            .filter(F.col("week_id").startswith(F.col("month_id").substr(1, 4))) 
                            .agg(F.max(F.col("week_id")).alias("max_week_december")).collect()[0]["max_week_december"]
        )

    d = date_dim.select('week_id').distinct()

    df_with_lag_lead = d.withColumn("lag_week_id", F.lag("week_id").over(Window.orderBy("week_id"))) \
                        .withColumn("lead_week_id", F.lead("week_id").over(Window.orderBy("week_id")))

    week_before = df_with_lag_lead.filter(F.col("week_id") == max_week_december).select("lag_week_id").first()[0]
    week_after = df_with_lag_lead.filter(F.col("week_id") == max_week_december).select("lead_week_id").first()[0]

    xmas_week_id = [week_before, max_week_december, week_after]

    time_of_day_df = time_of_day_df.withColumn('fest_flag',F.when(F.col('week_id').isin(xmas_week_id), 'XMAS')\
                                                        .when(F.col('month_id').cast('string').endswith('04'), 'APRIL')\
                                                        .otherwise('NONE'))
    
    conf_mapper["xmaxs_week_id"] = xmas_week_id
    files.conf_writer(conf_mapper, conf_path)

    # Last week of month (Payweey)
    last_sat = date_dim.filter(F.col('weekday_nbr') == 6).groupBy('month_id').agg(F.max('day_in_month_nbr').alias('day_in_month_nbr'))\
                                                    .withColumn('last_weekend_flag',F.lit('Y'))

    last_sat_df = date_dim.select('date_id', 'month_id', 'day_in_month_nbr')\
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
    r = flagged_df.withColumn('end_date',F.lit(conf_mapper["timeframe_end"]))\
            .withColumn('start_date',F.lit(conf_mapper["timeframe_start"]))\
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
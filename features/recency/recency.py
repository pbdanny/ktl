from pyspark.sql import functions as F
from datetime import datetime, timedelta


def day_regis(df, date_end):
    '''
    day register
        this method will map data into day register
    :param:
        df: dataframe
        date_end: data end date YYYY-MM-DD, e.g., 2022-09-04
    :return:
        df: dataframe with day register
    '''
    DATA_END_DATE = datetime.strptime(date_end, "%Y-%m-%d")
    df = df.withColumn("day_regis", F.datediff(
        F.lit(DATA_END_DATE), F.col("last_card_issue_date")))
    return df


def card_aging(df):
    '''
    card aging
        this method will map data into card aging
    :param:
        df: dataframe
    :return:
        df: dataframe with card aging

    '''
    df = df.withColumn("card_aging_day", F.datediff(
        F.col("first_contact_date"), F.col("last_card_issue_date")))
    return df


def start_date(df, recency_day):
    '''
    start date
        this method will map data into start date
        30, 90, 180, 360
    :param:
        recency_day: recency day e.g., [30, 90, 180, 360]
        df: dataframe
    :return:
        df: dataframe with start date

    '''
    for _day in recency_day:
        df = df.withColumn(f"r{_day}_start_date", F.date_add(
            F.col("first_contact_date"), -_day))
    return df


def recency_week(spark, cust_detail, recency_day: list, card_aging_day: int = 0):
    '''
    recency transaction
        this method will map data into recency transaction
        30, 90, 180, 360
    :param:
        recency_day: recency day e.g., [30, 90, 180, 360]
        cust_detail: customer detail
        card_aging_day: card aging day
    :return:
        cust_detail_wk: customer detail with recency transaction

    '''
    wk_dim = {}
    for i in recency_day:
        wk_dim[f"r{i}_wk_id"] = spark.table("tdm.v_date_dim").select(
            F.col("date_id").alias(f"r{i}_start_date"), F.col("week_id").alias(f"r{i}_wk_id"))
    for i in recency_day:
        cust_detail = cust_detail.join(
            wk_dim[f"r{i}_wk_id"], f"r{i}_start_date", "left")
    cust_detail_wk = cust_detail.where(
        F.col("card_aging_day") >= card_aging_day)
    return cust_detail_wk


def recency_txn(spark, sqlContext, recency_day: list, cust_fdbck_hh, date_end):
    '''
    recency transaction
        this method will map data into recency transaction
        30, 90, 180, 360
    :param:
        recency_day: recency day e.g., [30, 90, 180, 360]
        cust_fdbck_hh: customer feedback household
        date_end: end date of data
    :return:
        cust_detail_wk: customer detail with recency transaction
    '''
    hh_crd_issue_date = \
        (spark.table("tdm.v_customer_dim")
         .groupBy("household_id")
         .agg(F.max("card_issue_date").alias("last_card_issue_date"))
         .select("household_id", "last_card_issue_date")
         )
    date_dim = sqlContext.table("tdm.v_date_dim").select(
        "date_id", "week_id", "quarter_id", "period_id").withColumnRenamed("date_id", "first_contact_date")
    cust_detail = cust_fdbck_hh.join(
        hh_crd_issue_date, "household_id", "left").join(date_dim, "first_contact_date")
    cust_detail = card_aging(cust_detail)
    cust_detail = start_date(cust_detail, recency_day)
    cust_detail = day_regis(cust_detail, date_end)
    cust_detail_wk = recency_week(spark, cust_detail, recency_day)
    return cust_detail_wk

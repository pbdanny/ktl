import pyspark.sql.functions as F
from pyspark.sql import Window
from .logger import logger


@logger
def truprice(spark):
    '''
        truprice
        :param:
            spark, spark session
        return: dataframe
    '''
    truprice = spark.table("tdm_seg.srai_truprice_full_history").select(
        "household_id", "truprice_seg_desc", "period_id").drop_duplicates()
    return truprice


@logger
def facts(spark):
    '''
        facts
        :param:
            spark, spark session
        return: dataframe
    '''
    facts = spark.table("tdm_seg.srai_facts_full_history").select(
        "household_id", "facts_seg_desc", "week_id").drop_duplicates()
    return facts


@logger
def prefer_store(spark):
    '''
        prefer store
        :param:
            spark, spark session
        return: dataframe
    '''
    prefer_store = spark.table("tdm_seg.srai_prefstore_full_history").select(
        F.col("household_id").cast("bigint"), "pref_store_id", "pref_store_format", "pref_store_region", "period_id").drop_duplicates()
    return prefer_store


@logger
def lifestage(spark):
    '''
        lifestage
        :param:
            spark, spark session
        return: dataframe
    '''
    lifestage = spark.table('tdm.edm_lifestage_full').select(
        "household_id", "lifestage_seg_name", "mapping_quarter_id").withColumnRenamed("mapping_quarter_id", "quarter_id").drop_duplicates()
    return lifestage


@logger
def lifecycle(spark):
    '''
        lifecycle
        :param:
            spark, spark session
        return: dataframe
    '''
    wk_num = spark.table('tdm.v_date_dim').select("week_id", "week_num_sequence").drop_duplicates(
    ).withColumn("lead_week_num", F.col("week_num_sequence")+1)

    mapping_lead_week_id = \
        (wk_num.alias("a")
         .join(wk_num.alias("b"), on=[F.col("a.lead_week_num") == F.col("b.week_num_sequence")])
         .select("a.week_id", F.col("b.week_id").alias("mapping_week_id"))
         )

    lifecycle_desc = spark.read.csv(
        'dbfs:/FileStore/thanakrit/utils/lifecycle_detailed_name_mapping.csv', header=True, inferSchema=True)

    lifecycle = \
        (spark
         .table("tdm.srai_lifecycle_history")
         .select(F.col("household_id").cast("bigint"), "lifecycle_detailed_code", "week_id")
         .join(lifecycle_desc, "lifecycle_detailed_code")
         .join(mapping_lead_week_id, "week_id")
         .select("household_id", "lifecycle_name", "mapping_week_id")
         .withColumnRenamed("mapping_week_id", "week_id")
         )
    return lifecycle


def shopping_mission(spark, MIN_DATA_WEEK_ID, MAX_DATA_WEEK_ID):
    '''
        shopping_mission
        :param:
            spark, spark session
        return: dataframe
    '''
    shp_ms = \
        (spark
         .table("tdm_seg.shopping_missions_full_internal")
         .where(F.col("country") == "th")
         .where(F.col("week_id").between(MIN_DATA_WEEK_ID, MAX_DATA_WEEK_ID))
         .select("transaction_concat", "transaction_uid", "shopping_missions_id", "week_id")
         .drop_duplicates()
         )
    return shp_ms


@logger
def tender(spark, MIN_DATA_DATE, MAX_DATA_DATE):
    '''
        tender
        :param:
            spark, spark session
        return: dataframe
    '''
    tndr = \
        (spark.table('tdm.v_resa_group_resa_tran_tender')
         .where(F.col("country") == "th")
         .where(F.col("source") == "resa")
         .where(F.col("dp_data_dt").between(MIN_DATA_DATE, MAX_DATA_DATE))
         )
    return tndr


@logger
def single_tender(spark, MIN_DATA_DATE, MAX_DATA_DATE):
    '''
        single tender
        :param:
            spark, spark session
        return: dataframe
    '''
    single_tndr_typ = \
        (tender(spark, MIN_DATA_DATE, MAX_DATA_DATE)
         .withColumn("set_tndr_type", F.array_distinct(F.collect_list(F.col("tender_type_group")).over(Window.partitionBy("tran_seq_no"))))
         .withColumn("n_tndr_type", F.size(F.col("set_tndr_type")))
         .where(F.col("n_tndr_type") == 1)
         .withColumn("transaction_uid", F.concat_ws("_", F.col("tran_seq_no"), F.col("store"), F.col("dp_data_dt"),))
         .select("transaction_uid", "tender_type_group")
         .drop_duplicates()
         )
    return single_tndr_typ

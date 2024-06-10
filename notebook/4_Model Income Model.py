# Databricks notebook source
!pip install scorecardpy 
!pip install scikit-learn==0.22.1

# COMMAND ----------

import pandas as pd
import os
import numpy as np
import pickle
import scorecardpy as sc
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# file_path = '/Workspace/Users/kritawats.kraiwitchaicharoen@lotuss.com/Project/(Clone) KTL/Credit_scoring_model'
file_path = "/Volumes/prod/tdm_dev/edminput/filestore/model/ktl_income_model"

# COMMAND ----------

# DBTITLE 1,Load Data Set
# df_raw = spark.table("tdm_seg.kritawatkrai_th_year_rounded_full_agg_data")
df_raw = spark.table("tdm_dev.th_lotuss_ktl_txn_year_rounded_full_agg_data")

# COMMAND ----------

# DBTITLE 1,Dictionary Mapping Name
import re

new_col = df_raw.columns

# rename according to old features
replacements = {
    'DIV': 'L30',
    'DEP': 'L22',
    'SEC': 'L21',
    'CLASS': 'L20',
    '%3_14_50&3_51_416%': '0314050',
    '%2_3_195&2_3_52%': '0233701',
    '%13_77&13_78&13_79%': '1305',
    '%1_2_187&1_2_86%': '0102016',
    'Total_Spend': 'SPEND',
    'Total_Units': 'UNITS',
    'Total_Visits': 'VISITS',
    'MINISUPERMARKET': 'EXPRESS',
    'BKK&VICINITIES': 'BKK',
    '%1_71_136%': '0101235',
    '%2_66_353%': '0233707',
    '%1_1_149%': '0101234',
    '%13_67%': '1334'
    
}
replaced_col = list(map(lambda s: re.sub('|'.join(re.escape(key) for key in replacements.keys()), lambda match: replacements[match.group(0)], s), new_col))

# transform category code to old format
transformed_list = []

for string in replaced_col:
    parts = string.split('%')
    
    if len(parts) > 1:
        parts_between_percent = parts[1].split('_')
        
        for i in range(len(parts_between_percent)):
            if len(parts_between_percent[i]) == 1:
                parts_between_percent[i] = '0' + parts_between_percent[i]
        
        if len(parts_between_percent) >= 3 and len(parts_between_percent[2]) < 3:
            parts_between_percent[2] = '0' * (3 - len(parts_between_percent[2])) + parts_between_percent[2]
        
        transformed_string = parts[0] + ''.join(parts_between_percent)  + ''.join(parts[2:])
        transformed_list.append(transformed_string)
    else:
        transformed_list.append(string)


# Create mapping dictionary
dic = dict(zip(transformed_list, new_col))


# COMMAND ----------

# DBTITLE 1,Functions
def replace_missing_value(df, missing_tab):
    for i in missing_tab.index:
        df[i] = df[i].fillna(-9999999)

#----------------------------------------------------------------------------------------------

def calcScore(proba):
    E = 497
    D = 46
    P = proba
    scores = E + (D/np.log(2))*np.log((1-P)/P)
    return scores

#----------------------------------------------------------------------------------------------

def EncodeLabel(df):
    label_encoder = preprocessing.LabelEncoder()
     # Iterate through each column
    for i, col in enumerate(df):
        if df[col].dtype == 'object' and col != id_column:
        # Map the categorical features to integers
           df[col] = label_encoder.fit_transform(np.array(df[col].astype(str)).reshape((-1,)))
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model ID: M001
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Features
selected_features = ['CC_TENURE',
 'PCT_TIME_MORNING_VISITS',
 'N_STORES',
 'PCT_TIME_EVENING_MARK_DOWN_SPEND',
 'PCT_SF_SUPERMARKET_VISITS',
 'PCT_CAT_L21_0203004_VISITS',
 'N_DISTINCT_L20',
 'PCT_CAT_L21_0233701_UNITS',
 'PCT_CAT_L21_1011072_UNITS',
 'PCT_PYMT_CARD_VISITS',
 'PCT_CAT_L21_0139624_VISITS',
 'PCT_CAT_L21_0138042_UNITS',
 'PCT_CAT_L22_0139_VISITS',
 'CAT_L22_1334_L3_SPU',
 'L3_PROMO_ITEMS',
 'PCT_TIME_EVENING_VISITS',
 'CAT_L30_10_L6_UPV',
 'PCT_WKND_FLAG_N_SPEND',
 'CAT_L22_0138_L9_UPV',
 'MAX_VISITS_MNTH',
 'PCT_CAT_L22_1011_L3_VISITS',
 'MAX_SPEND_MNTH',
 'PCT_CAT_L22_1306_L3_VISITS',
 'PCT_SF_EXPRESS_SPEND',
 'CAT_L30_10_SPV',
 'PCT_CAT_L22_1046_L9_UNITS',
 'PCT_CAT_L21_0233704_UNITS',
 'PCT_CAT_L22_1334_L3_SPEND',
 'PCT_CAT_L21_0203004_UNITS',
 'CAT_L22_0204_SPU',
 'PCT_WKND_FLAG_Y_VISITS',
 'PCT_CAT_L21_0309045_SPEND',
 'CAT_L30_13_L3_SPU',
 'PCT_CAT_L22_1305_L6_UNITS',
 'PCT_CAT_L22_1011_L6_SPEND',
 'PCT_CAT_L21_1306011_SPEND',
 'PCT_WKND_FLAG_Y_UNITS',
 'PCT_CAT_L21_0102070_SPEND']

# COMMAND ----------

filtered_dic = {key: value for key, value in dic.items() if key in selected_features}
len(filtered_dic)

# COMMAND ----------

# DBTITLE 1,Check Missing Columns
dict_values = list(filtered_dic.keys())

# Iterate over the list and check if the item is not in the dictionary values
for item in selected_features:
    if item not in dict_values:
        print(item)

# COMMAND ----------

df_oot = df_raw.select(list(filtered_dic.values()))


# COMMAND ----------

for new_name, old_name in filtered_dic.items():
    df_oot = df_oot \
    .withColumnRenamed(old_name, new_name)

# COMMAND ----------

df_oot = df_oot.toPandas()

# COMMAND ----------

# DBTITLE 1,Check Missing Value
# Calculate the percentage of missing values
df_temp = df_oot.copy()
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

# DBTITLE 1,Transform Data
xoot = df_oot[selected_features]

# COMMAND ----------

# DBTITLE 1,WOE
bins = np.load(file_path + r'/M001_bin.npy',allow_pickle=True).item()

oot_woe = sc.woebin_ply(xoot, bins)
oot_woe.head()

# COMMAND ----------

oot_woe.columns  = [x.upper() for x in oot_woe.columns]
oot_woe.shape

# COMMAND ----------

# DBTITLE 1,Final Features
selected_features = ['CC_TENURE_WOE',
 'PCT_TIME_MORNING_VISITS_WOE',
 'N_STORES_WOE',
 'PCT_SF_SUPERMARKET_VISITS_WOE',
 'PCT_TIME_EVENING_MARK_DOWN_SPEND_WOE',
 'PCT_CAT_L21_0233701_UNITS_WOE',
 'PCT_CAT_L21_0203004_VISITS_WOE',
 'PCT_CAT_L21_1011072_UNITS_WOE',
 'PCT_CAT_L22_0139_VISITS_WOE',
 'CAT_L22_0204_SPU_WOE',
 'PCT_CAT_L22_1334_L3_SPEND_WOE',
 'PCT_CAT_L21_0309045_SPEND_WOE',
 'PCT_CAT_L21_0138042_UNITS_WOE',
 'PCT_PYMT_CARD_VISITS_WOE',
 'PCT_CAT_L22_1046_L9_UNITS_WOE',
 'PCT_CAT_L21_0139624_VISITS_WOE',
 'PCT_CAT_L21_0203004_UNITS_WOE',
 'PCT_WKND_FLAG_Y_UNITS_WOE',
 'PCT_TIME_EVENING_VISITS_WOE',
 'N_DISTINCT_L20_WOE',
 'CAT_L30_10_L6_UPV_WOE',
 'PCT_CAT_L21_0233704_UNITS_WOE',
 'PCT_SF_EXPRESS_SPEND_WOE',
 'PCT_WKND_FLAG_N_SPEND_WOE',
 'CAT_L30_13_L3_SPU_WOE',
 'CAT_L30_10_SPV_WOE',
 'CAT_L22_0138_L9_UPV_WOE',
 'PCT_CAT_L22_1011_L3_VISITS_WOE',
 'CAT_L22_1334_L3_SPU_WOE',
 'MAX_SPEND_MNTH_WOE',
 'PCT_WKND_FLAG_Y_VISITS_WOE',
 'MAX_VISITS_MNTH_WOE',
 'PCT_CAT_L22_1011_L6_SPEND_WOE',
 'PCT_CAT_L21_0102070_SPEND_WOE',
 'PCT_CAT_L22_1306_L3_VISITS_WOE',
 'L3_PROMO_ITEMS_WOE',
 'PCT_CAT_L21_1306011_SPEND_WOE',
 'PCT_CAT_L22_1305_L6_UNITS_WOE']

# COMMAND ----------

pkl_filename = file_path + r'/M001_final_model.pkl'
model_1 = pickle.load(open(pkl_filename, 'rb'))

# COMMAND ----------

xoot_select  = oot_woe[selected_features]
xoot_select.head(3)

# COMMAND ----------

test_oot_pred = model_1.predict_proba(xoot_select)[:,1]
xoot_select['Probability'] = test_oot_pred
xoot_select['Score'] = calcScore(test_oot_pred)
selected_features.append('Probability')
selected_features.append('Score')

# COMMAND ----------

# xoot_select.head()
results = spark.createDataFrame(xoot_select)
# results.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_income_1_prediction")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model ID: M002
# MAGIC

# COMMAND ----------

selected_features = [
'MAX_VISITS_MNTH',
 'PCT_WKND_FLAG_Y_UNITS',
 'PCT_PYMT_CARD_VISITS',
 'PCT_CAT_L22_1046_VISITS',
 'PCT_CAT_L21_0233704_SPEND',
 'N_STORES',
 'CC_TENURE',
 'CAT_L22_1334_L3_SPU',
 'SD_VISITS_MNTH',
 'PCT_WKND_FLAG_Y_VISITS',
 'PCT_TIME_LATE_MARK_DOWN_UNITS',
 'PCT_TIME_MORNING_VISITS',
 'PCT_TIME_LATE_MARK_DOWN_SPEND',
 'PCT_WKND_FLAG_Y_SPEND',
 'PCT_CAT_L21_1011072_SPEND',
 'PCT_CAT_L22_1046_L9_UNITS',
 'PCT_CAT_L21_0233701_SPEND',
 'PCT_SF_EXPRESS_SPEND',
 'PCT_SF_SUPERMARKET_VISITS',
 'PCT_CAT_L21_0309045_VISITS',
 'PCT_CAT_L21_0233704_VISITS',
 'PCT_CAT_L22_1334_L3_SPEND',
 'PCT_CAT_L22_1334_L3_UNITS',
 'PCT_CAT_L21_1011072_UNITS',
 'L9_MARK_DOWN_ITEMS',
 'PCT_CAT_L22_1046_L3_SPEND',
 'PCT_CAT_L21_0138042_UNITS',
 'SD_UNITS_MNTH',
 'PCT_CAT_L22_1046_SPEND',
 'PCT_CAT_L21_0233704_UNITS',
 'CAT_L22_1334_L3_UPV',
 'PCT_CAT_L21_0203004_VISITS',
 'PCT_CAT_L21_0233701_UNITS',
 'CAT_L22_1334_UPV',
 'PCT_TIME_EVENING_MARK_DOWN_UNITS',
 'CAT_L30_02_SPU',
 'PCT_PCT_LAST_WKND_UNITS',
 'PCT_MARK_DOWN_UNITS',
 'PCT_CAT_L22_1046_L9_SPEND']

# COMMAND ----------

filtered_dic = {key: value for key, value in dic.items() if key in selected_features}
len(filtered_dic)


# COMMAND ----------

# DBTITLE 1,Check Missing Columns
dict_values = list(filtered_dic.keys())

# Iterate over the list and check if the item is not in the dictionary values
for item in selected_features:
    if item not in dict_values:
        print(item)

# COMMAND ----------

df_oot = df_raw.select(list(filtered_dic.values()))
# print(df_oot.shape)

# COMMAND ----------

for new_name, old_name in filtered_dic.items():
    df_oot = df_oot \
    .withColumnRenamed(old_name, new_name)

# COMMAND ----------

df_oot = df_oot.toPandas()

# COMMAND ----------

# Calculate the percentage of missing values
df_temp = df_oot.copy()
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

replace_missing_value(df_oot, missing_values_table)

# COMMAND ----------

# Calculate the percentage of missing values
df_temp = df_oot
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

# DBTITLE 1,Transform Data
xoot = df_oot[selected_features]


# COMMAND ----------

# DBTITLE 1,Encoding
xoot_select = EncodeLabel(xoot)
xoot_select.shape

# COMMAND ----------

# DBTITLE 1,Prediction
pkl_filename = file_path + r'/M002_final_model.pkl'
model_2 = pickle.load(open(pkl_filename, 'rb'))

# COMMAND ----------

xoot_select = xoot_select[selected_features]
xoot_select.head(3)

# COMMAND ----------

test_oot_pred = model_2.predict_proba(xoot_select)[:,1]
xoot_select['Probability'] = test_oot_pred
xoot_select['Score'] = calcScore(test_oot_pred)
selected_features.append('Probability')
selected_features.append('Score')

# COMMAND ----------

# xoot_select.head()
results = spark.createDataFrame(xoot_select) 
# results.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_income_2_prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model ID: M003
# MAGIC

# COMMAND ----------

selected_features = ['N_STORES',
 'CC_TENURE',
 'PCT_CAT_L22_0139_UNITS',
 'PCT_TIME_EVENING_VISITS',
 'PCT_CAT_L21_0233701_SPEND',
 'PCT_PYMT_CARD_VISITS',
 'PCT_CAT_L22_1334_L3_SPEND',
 'SD_SPEND_MNTH',
 'PCT_CAT_L21_1011072_UNITS',
 'CAT_L22_0309_UPV',
 'PCT_TIME_LATE_UNITS',
 'PCT_CAT_L21_0203004_UNITS',
 'N_DISTINCT_L20',
 'PCT_CAT_L21_0101234_VISITS',
 'PCT_CAT_L22_0115_VISITS',
 'PCT_TIME_LATE_VISITS',
 'PCT_WKND_FLAG_Y_UNITS',
 'PREMIUM_SPV',
 'PCT_WKND_FLAG_Y_SPEND',
 'PCT_TIME_EVENING_MARK_DOWN_SPEND',
 'PCT_CAT_L21_0233704_UNITS',
 'PCT_CAT_L21_0309045_UNITS',
 'CAT_L30_10_L6_SPV',
 'PCT_SF_SUPERMARKET_VISITS',
 'PCT_TIME_MORNING_UNITS',
 'PCT_CAT_L22_0116_SPEND',
 'PCT_CAT_L21_0139624_VISITS',
 'CAT_L22_0412_SPV',
 'CAT_L22_1334_UPV',
 'PCT_CAT_L21_0204222_UNITS',
 'PCT_PYMNT_COUPON_VISITS',
 'L3_PROMO_ITEMS',
 'PCT_CAT_L21_0138042_SPEND',
 'PCT_TIME_MORNING_PROMO_SPEND',
 'CAT_L30_02_SPU',
 'PCT_TIME_MORNING_SPEND',
 'PCT_CAT_L21_1046020_SPEND',
 'PCT_CAT_L21_0139221_VISITS',
 'PCT_CAT_L21_1046243_UNITS',
 'CAT_L22_1305_SPU',
 'Q2_UPV',
 'PCT_CAT_L21_0102070_SPEND',
 'PCT_CAT_L21_0138042_UNITS',
 'PCT_CAT_L22_1046_L3_SPEND',
 'CAT_L22_0101_SPV',
 'Q1_UPV',
 'PCT_CAT_L22_0116_L9_SPEND',
 'PCT_CAT_L21_1306012_UNITS',
 'PCT_CAT_L22_1046_L9_UNITS',
 'PCT_CAT_L21_0102070_UNITS',
 'Q4_UPV',
 'PCT_CAT_L22_0204_SPEND',
 'CAT_L22_1305_L6_SPU',
 'PCT_CAT_L30_10_UNITS',
 'CAT_L22_1046_SPV',
 'Q3_UPV',
 'PCT_CAT_L30_10_L9_SPEND',
 'PCT_MARK_DOWN_SPEND',
 'PCT_CAT_L22_1046_VISITS']

# COMMAND ----------

filtered_dic = {key: value for key, value in dic.items() if key in selected_features}
len(filtered_dic)


# COMMAND ----------

dict_values = list(filtered_dic.keys())

# Iterate over the list and check if the item is not in the dictionary values
for item in selected_features:
    if item not in dict_values:
        print(item)

# COMMAND ----------

df_oot = df_raw.select(list(filtered_dic.values()))
# print(df_oot.shape)

# COMMAND ----------

for new_name, old_name in filtered_dic.items():
    df_oot = df_oot \
    .withColumnRenamed(old_name, new_name)

# COMMAND ----------

df_oot = df_oot.toPandas()

# COMMAND ----------

# DBTITLE 1,Check Missing Value
# Calculate the percentage of missing values
df_temp = df_oot.copy()
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

# DBTITLE 1,Replace Missing Value
replace_missing_value(df_oot, missing_values_table)


# COMMAND ----------

# Calculate the percentage of missing values
df_temp = df_oot
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

# DBTITLE 1,Transform Data
xoot = df_oot[selected_features]

# COMMAND ----------

# DBTITLE 1,WOE
bins = np.load(file_path + r'/M003_bin.npy',allow_pickle=True).item()

oot_woe = sc.woebin_ply(xoot, bins)
oot_woe.head()

# COMMAND ----------

oot_woe.columns  = [x.upper() for x in oot_woe.columns]
oot_woe.shape

# COMMAND ----------

# DBTITLE 1,Final Features
selected_features = ['N_STORES_WOE',
 'CC_TENURE_WOE',
 'PCT_CAT_L22_0139_UNITS_WOE',
 'PCT_TIME_EVENING_VISITS_WOE',
 'PCT_CAT_L21_0233701_SPEND_WOE',
 'PCT_PYMT_CARD_VISITS_WOE',
 'PCT_CAT_L22_1334_L3_SPEND_WOE',
 'PCT_CAT_L21_1011072_UNITS_WOE',
 'PCT_TIME_LATE_UNITS_WOE',
 'SD_SPEND_MNTH_WOE',
 'PCT_CAT_L21_0203004_UNITS_WOE',
 'N_DISTINCT_L20_WOE',
 'PCT_CAT_L22_0204_SPEND_WOE',
 'PCT_CAT_L21_0101234_VISITS_WOE',
 'CAT_L22_0309_UPV_WOE',
 'PCT_CAT_L22_0115_VISITS_WOE',
 'PCT_TIME_LATE_VISITS_WOE',
 'PCT_WKND_FLAG_Y_UNITS_WOE',
 'PCT_WKND_FLAG_Y_SPEND_WOE',
 'PCT_TIME_EVENING_MARK_DOWN_SPEND_WOE',
 'PREMIUM_SPV_WOE',
 'PCT_CAT_L21_0309045_UNITS_WOE',
 'PCT_CAT_L21_0233704_UNITS_WOE',
 'CAT_L30_10_L6_SPV_WOE',
 'PCT_CAT_L22_0116_SPEND_WOE',
 'PCT_TIME_MORNING_UNITS_WOE',
 'PCT_SF_SUPERMARKET_VISITS_WOE',
 'PCT_CAT_L21_0139624_VISITS_WOE',
 'CAT_L30_02_SPU_WOE',
 'L3_PROMO_ITEMS_WOE',
 'CAT_L22_1334_UPV_WOE',
 'PCT_CAT_L21_0138042_SPEND_WOE',
 'CAT_L22_0412_SPV_WOE',
 'PCT_TIME_MORNING_PROMO_SPEND_WOE',
 'PCT_PYMNT_COUPON_VISITS_WOE',
 'PCT_TIME_MORNING_SPEND_WOE',
 'PCT_CAT_L21_1046020_SPEND_WOE',
 'PCT_CAT_L21_0139221_VISITS_WOE',
 'PCT_CAT_L21_1046243_UNITS_WOE',
 'CAT_L22_1305_SPU_WOE',
 'Q2_UPV_WOE',
 'PCT_CAT_L21_0102070_SPEND_WOE',
 'PCT_CAT_L21_0204222_UNITS_WOE',
 'PCT_CAT_L21_0138042_UNITS_WOE',
 'PCT_CAT_L22_1046_L3_SPEND_WOE',
 'CAT_L22_0101_SPV_WOE',
 'PCT_CAT_L22_0116_L9_SPEND_WOE',
 'Q1_UPV_WOE',
 'Q4_UPV_WOE',
 'PCT_CAT_L21_0102070_UNITS_WOE',
 'PCT_CAT_L22_1046_L9_UNITS_WOE',
 'PCT_CAT_L21_1306012_UNITS_WOE',
 'PCT_CAT_L30_10_UNITS_WOE',
 'CAT_L22_1046_SPV_WOE',
 'CAT_L22_1305_L6_SPU_WOE',
 'Q3_UPV_WOE',
 'PCT_CAT_L30_10_L9_SPEND_WOE',
 'PCT_MARK_DOWN_SPEND_WOE',
 'PCT_CAT_L22_1046_VISITS_WOE']

# COMMAND ----------

# DBTITLE 1,Prediction
pkl_filename = file_path + r'/M003_final_model.pkl'

model_3 = pickle.load(open(pkl_filename, 'rb'))

# COMMAND ----------

xoot_select  = oot_woe[selected_features]
xoot_select.head(3)

# COMMAND ----------

test_oot_pred = model_3.predict_proba(xoot_select)[:,1]
xoot_select['Probability'] = test_oot_pred
xoot_select['Score'] = calcScore(test_oot_pred)
selected_features.append('Probability')
selected_features.append('Score')

# COMMAND ----------

# xoot_select.head()
results = spark.createDataFrame(xoot_select) 
results.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_income_3_prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model ID: M004
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Final Features
selected_features = [
 'PCT_PYMT_CARD_VISITS',
 'N_STORES',
 'SD_VISITS_MNTH',
 'PCT_PYMNT_COUPON_VISITS',
 'CC_TENURE',
 'PCT_TIME_LATE_MARK_DOWN_UNITS',
 'PCT_WKND_FLAG_N_SPEND',
 'PCT_CAT_L21_0309045_UNITS',
 'PCT_TIME_LATE_MARK_DOWN_SPEND',
 'VISITS',
 'PCT_CAT_L22_1046_VISITS',
 'PCT_CAT_L22_1046_UNITS',
 'PCT_TIME_LATE_VISITS',
 'PCT_CAT_L21_0138042_UNITS',
 'CAT_L22_1334_L6_UPV',
 'PCT_PCT_PREMIUM_VISITS',
 'L9_MARK_DOWN_ITEMS',
 'PCT_MARK_DOWN_UNITS',
 'PCT_CAT_L21_1011072_VISITS',
 'PCT_CAT_L22_1334_L3_SPEND',
 'PCT_CAT_L21_0233704_SPEND',
 'PCT_CAT_L21_1011072_UNITS',
 'SD_UNITS_MNTH',
 'PCT_CAT_L21_0139624_VISITS',
 'PCT_CAT_L22_0116_UNITS',
 'PCT_CAT_L21_1046020_UNITS',
 'PCT_CAT_L21_1011072_SPEND']

# COMMAND ----------

filtered_dic = {key: value for key, value in dic.items() if key in selected_features}
len(filtered_dic)


# COMMAND ----------

dict_values = list(filtered_dic.keys())

# Iterate over the list and check if the item is not in the dictionary values
for item in selected_features:
    if item not in dict_values:
        print(item)

# COMMAND ----------

df_oot = df_raw.select(list(filtered_dic.values()))
# print(df_oot.shape)

# COMMAND ----------

for new_name, old_name in filtered_dic.items():
    df_oot = df_oot \
    .withColumnRenamed(old_name, new_name)

# COMMAND ----------

df_oot = df_oot.toPandas()

# COMMAND ----------

# DBTITLE 1,Check Missing Value
# Calculate the percentage of missing values
df_temp = df_oot.copy()
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

# DBTITLE 1,Replace Missing Value
replace_missing_value(df_oot, missing_values_table)

# COMMAND ----------

# Calculate the percentage of missing values
df_temp = df_oot
missing_values = df_temp.isnull().sum()
missing_values_percentage = 100 * missing_values/len(df_temp)
missing_values_table = pd.concat([missing_values, missing_values_percentage], axis = 1)
missing_values_table = missing_values_table.rename(columns = {0:'Missing Values', 1:'% of Missing Values'})
missing_values_table = missing_values_table[missing_values_table.iloc[:, 0] != 0].sort_values(by = '% of Missing Values',
                       ascending = False).round(2)
missing_values_table

# COMMAND ----------

# DBTITLE 1,Transform Data
xoot = df_oot[selected_features]


# COMMAND ----------

# DBTITLE 1,Encoding
xoot_select = EncodeLabel(xoot)
xoot_select.shape

# COMMAND ----------

# DBTITLE 1,Prediction
pkl_filename = file_path + r'/M004_final_model.pkl'
model_4 = pickle.load(open(pkl_filename, 'rb'))

# COMMAND ----------

xoot_select  = xoot_select[selected_features]
xoot_select.head(3)

# COMMAND ----------

test_oot_pred = model_4.predict_proba(xoot_select)[:,1]
xoot_select['Probability'] = test_oot_pred
xoot_select['Score'] = calcScore(test_oot_pred)
selected_features.append('Probability')
selected_features.append('Score')

# COMMAND ----------

# xoot_select.head()
results = spark.createDataFrame(xoot_select) 
results.write.mode("overwrite").saveAsTable("tdm_seg.kritawatkrai_th_income_4_prediction")


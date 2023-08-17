# Databricks notebook source
# import os

# COMMAND ----------

# from utils.data import Data

# COMMAND ----------

# spark.sparkContext.setCheckpointDir('dbfs:/FileStore/niti/temp/checkpoint') # must set checkpoint before passing txnItem

# COMMAND ----------

# data = Data(spark, "202307_model", False)

# COMMAND ----------

# data.feature_mapper

# COMMAND ----------

# data.get_features("cust_details").display()

# COMMAND ----------

# data.get_features("cust_tender_kpi").display()

# COMMAND ----------

# data.get("credit").display()

# COMMAND ----------

# data.get("cash").display()

# COMMAND ----------

# data.get("credit")

# COMMAND ----------

# card_df = data.get("credit").toPandas().set_index("household_id").drop(["cash_card_customer_flag"], axis=1)

# COMMAND ----------

# card_df

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # mlflow

# COMMAND ----------

from utils.data import Data

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_curve, RocCurveDisplay, recall_score, f1_score, auc, average_precision_score, precision_recall_curve, PrecisionRecallDisplay

from hyperopt.pyll import scope
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, SparkTrials

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

# COMMAND ----------

# get feature
sel_col = ['prop_visit_by_cash_360_days', 'prop_visit_by_card_360_days', 'prop_promo_sales', 'afterlunch_prop_visits_360_days', 'early_morning_prop_visits_360_days', 'evening_prop_visits_360_days', 'late_evening_prop_visits_360_days', 'lunch_prop_visits_360_days', 'morning_prop_visits_360_days', 'night_prop_visits_360_days', 'Weekday_prop_visits_360_days', 'Weekend_prop_visits_360_days', 'total_sales_360_days', 'total_visits_360_days', 'apparel_prop_sales', 'baby_prop_sales', 'baby_kids_prop_sales', 'babyaccessories_prop_sales', 'babysupplement_prop_sales', 'bakery_prop_sales', 'bakeryboughtin_prop_sales', 'ball_prop_sales', 'basicfood_prop_sales', 'beveragesparty_prop_sales', 'beveragesparty_prop_visits', 'breakfast_icecream_prop_sales', 'breakfast_icecream_prop_visits', 'cookameal_prop_sales', 'cookameal_prop_visits', 'cosmetics_prop_sales', 'dairy_snacksbreaktime_prop_sales', 'dairy_snacksbreaktime_prop_visits', 'delicatessen_prop_sales', 'fishseafood_prop_sales', 'foodcourt_prop_sales', 'foodsupplement_prop_sales', 'freshmeat_seafood_prop_sales', 'frozen_prop_sales', 'fruit_prop_sales', 'fruits_healthy_prop_sales', 'giftgrocery_prop_sales', 'haircare_prop_sales', 'health_prod_prop_sales', 'health_prod_prop_visits', 'hle_prop_sales', 'hlh_prop_sales', 'hlh_prop_visits', 'hotbevsugar_prop_sales', 'household_hlh_prop_sales', 'household_hlh_prop_visits',
           'icecreamdessert_prop_sales', 'kid_prop_sales', 'kitchenstock_precooked_prop_sales', 'kitchenstock_precooked_prop_visits', 'leisure_prop_sales', 'luggage_prop_sales', 'mealsolutions_prop_sales', 'otherbakery_prop_sales', 'otherdairydrinks_prop_sales', 'otherhh_prop_sales', 'otherliquor_prop_sales', 'othermeat_seafoods_prop_sales', 'otherprecooked_prop_sales', 'othertobacco_prop_sales', 'othertoiletries_prop_sales', 'othertoiletries_prop_units', 'papergoods_prop_sales', 'personalcare_prop_sales', 'personalcare_prop_visits', 'petfoodsupplys_prop_sales', 'petshopstandalone_prop_sales', 'pharmacy_prop_sales', 'processedfood_prop_sales', 'rtd_prop_sales', 'skincare_prop_sales', 'ConsumptionforToday_prop_visits_360_days', 'Emergency_prop_visits_360_days', 'ImmediateConsumption_prop_visits_360_days', 'Others_prop_visits_360_days', 'StockUp_prop_visits_360_days', 'TopUp_prop_visits_360_days', 'Weekly_prop_visits_360_days', 'trader_prop_visits_360_days', 'online_all_sales_360_days', 'online_all_sales_90_days', 'online_all_sales_30_days', 'offline_hde_sales_360_days', 'offline_hde_visits_360_days', 'offline_hde_sales_180_days', 'offline_hde_sales_30_days', 'offline_gofresh_sales_360_days', 'offline_gofresh_sales_30_days', 'card_aging_day'] + ['truprice_seg_desc', 'facts_seg_desc', 'lifestage_seg_name', 'lifecycle_name'] + ["household_id", "Bought_Status"]


def get_train_test_data(data, target, train_size=0.8, test_size=0.5):
    """
    Split data into train, validation and test sets
    :param:
        data: pandas dataframe
        target: dictionary with target column name and positive label
        train_size: float, size of train set
        test_size: float, size of test set
    :return:
        X_train: pandas dataframe, train set features
        X_val: pandas dataframe, validation set features
        X_test: pandas dataframe, test set features
        y_train: numpy array, train set target
        y_val: numpy array, validation set target
        y_test: numpy array, test set target
    """
    X = data.drop(target["column"], axis=1)
    y = data[[target["column"]]].assign(target=lambda x: np.where(
        x == target["positive"], 1, 0))["target"].to_numpy()

    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=train_size, random_state=0, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=test_size, random_state=0, stratify=y_rem)
    return X_train, X_val, X_test, y_train, y_val, y_test

# COMMAND ----------

# prjct_nm = "202307_model"
# test = False
# payment = "credit"
# tmp_data = Data(spark, prjct_nm, test)
# tmp_df = tmp_data.get(cash_card_customer_flag=payment)
# tmp_df = tmp_df.drop("cash_card_customer_flag")
# tmp_df = tmp_df.select(sel_col)
# tmp_df = tmp_df.toPandas().set_index("household_id")
# # Onehot categorical variables
# data = pd.get_dummies(tmp_df,
#                       prefix=["trprce", "fcts", "lfstg", "lfcyc"],
#                       columns=["truprice_seg_desc", "facts_seg_desc", "lifestage_seg_name", "lifecycle_name"])
# del [tmp_data, tmp_df]

# COMMAND ----------

# data

# COMMAND ----------

# data[["Bought_Status"]].value_counts()

# COMMAND ----------

prjct_nm = "202307_model"
test = False
payment = "cash"
tmp_data = spark.read.parquet(f"abfss://data@pvtdmdlsazc02.dfs.core.windows.net/tdm_seg.db/niti/lmp/insurance_lead/{prjct_nm}/features/all_feature.parquet")

# COMMAND ----------

tmp_data

# COMMAND ----------

# tmp_df = tmp_data.get(cash_card_customer_flag=payment)
tmp_df = tmp_data.filter(F.col("cash_card_customer_flag") == payment)
tmp_df = tmp_df.drop("cash_card_customer_flag")
tmp_df = tmp_df.select(sel_col)
tmp_df = tmp_df.toPandas().set_index("household_id")
# Onehot categorical variables
data = pd.get_dummies(
    tmp_df,
    prefix=["trprce", "fcts", "lfstg", "lfcyc"],
    columns=[
        "truprice_seg_desc",
        "facts_seg_desc",
        "lifestage_seg_name",
        "lifecycle_name",
    ],
)
del [tmp_data, tmp_df]

# COMMAND ----------

data = data.drop_duplicates()

# COMMAND ----------

data.columns = data.columns.str.replace("\W", "_", regex=True)

# COMMAND ----------

data.groupby("Bought_Status").count()

# COMMAND ----------

# Replace space with underscore
target = {
    "column": "Bought_Status",
    "positive": "Bought",
}
X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_data(
    data, target=target)

# COMMAND ----------

X_train.columns

# COMMAND ----------

positive = X_train[y_train==1]
negative = X_train[y_train==0]

# COMMAND ----------

selected_crit = int(0.9 * len(positive) // 1)

# COMMAND ----------

positive_sample = positive.sample(selected_crit)
negative_sample = negative.sample(selected_crit)

# COMMAND ----------

join_data = pd.concat([positive_sample, negative_sample], ignore_index=True, sort=False)

# COMMAND ----------

tmp_y = [1]*selected_crit + [0]*selected_crit

# COMMAND ----------

join_data["target"] = tmp_y

# COMMAND ----------

join_data = join_data.sample(frac = 1)

# COMMAND ----------

X_train_bal = join_data.drop("target", axis=1)
y_train_bal = join_data["target"].to_numpy()

# COMMAND ----------

print(len(X_train))
print(len(X_train_bal))
print(len(X_val))
print(len(X_test))

# COMMAND ----------

def get_plot_metrics(model, X_test, y_test):

    # predit
    pred = model.predict(X_test)
    pred_prop = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    g = ConfusionMatrixDisplay(cm, model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    g.plot(ax=ax, values_format='')

    # balance accuracy
    bal_acc = balanced_accuracy_score(y_test, pred)

    # auc
    fpr, tpr, thresholds = roc_curve(y_test, pred_prop)
    roc_auc = auc(fpr, tpr)
    g = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="")
    fig, ax = plt.subplots(figsize=(8, 6))
    g.plot(ax=ax)

    # precision-recall, F1
    rec_scr = recall_score(y_test, pred)

    f1_scr = f1_score(y_test, pred)

    avg_prc = average_precision_score(y_test, pred_prop)

    precision, recall, thresholds = precision_recall_curve(y_test, pred_prop)
    g = PrecisionRecallDisplay(
        precision=precision, recall=recall, average_precision=avg_prc, estimator_name=""
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    g.plot(ax=ax)

    # Geometric mean score
    print(f"Balance accuracy score : {bal_acc:.4f}")
    print(f"Average Precision score : {avg_prc:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Recall Score : {rec_scr:.4f}")
    print(f"F1 score : {f1_scr:.4f}")
    print(f"Geometric Mean score : {geometric_mean_score(y_test, pred):.4f}")

    # Show the classification report
    print(classification_report_imbalanced(y_test, pred))

    return bal_acc, roc_auc, rec_scr, f1_scr, avg_prc

# COMMAND ----------

experiment_path = "/Users/niti.buesamae@lotuss.com/LMP/Lead_Generation/EDM_202307/model_02A_cash/imbalance_exp"
mlflow.set_experiment(experiment_path)

# COMMAND ----------

# search_space = {
#     'n_estimators': scope.int(hp.quniform("n_estimators", 100, 400, 50)),
#     'max_depth': scope.int(hp.quniform("max_depth", 4, 24, 2)),
#     'max_features': scope.int(hp.quniform("max_features", 4, 24, 2)),
#     'min_samples_leaf': scope.int(hp.quniform("min_samples_leaf", 4, 24, 2)),
#     'max_samples': scope.float(hp.uniform("max_samples", 0.1, 1)),
# }

# model_name = "balance_random_forest"

# def train_model(params):
    
#     X_train, y_train, X_val, y_val = broadcast_data.value
    
#     with mlflow.start_run(nested=True):
#       model = BalancedRandomForestClassifier(**params, random_state=0)
#       model.fit(X_train, y_train)
      
#       # Get param from fitted model
#       fitted_param = model.get_params()
      
#       pred = model.predict(X_val)
#       pred_prop = model.predict_proba(X_val)[:,1]
      
#       bal_acc = balanced_accuracy_score(y_val, pred)
#       fpr, tpr, thresholds = roc_curve(y_val, pred_prop)
#       roc_auc = auc(fpr, tpr)
      
#       rec_scr = recall_score(y_val, pred)
#       f1_scr = f1_score(y_val, pred)
#       avg_prc = average_precision_score(y_val, pred_prop)

#       mlflow.log_metric('val_bal_acc', bal_acc)
#       mlflow.log_metric('val_roc_auc', roc_auc)
#       mlflow.log_metric('val_rec_scr', rec_scr)
#       mlflow.log_metric('val_f1_scr', f1_scr)
#       mlflow.log_metric('val_avg_prc', avg_prc)

#       loss = -1.0*avg_prc
        
#       return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

search_space = {
    'n_estimators': scope.int(hp.quniform("n_estimators", 100, 400, 50)),
    'max_depth': scope.int(hp.quniform("max_depth", 4, 24, 2)),
    'max_features': scope.int(hp.quniform("max_features", 4, 24, 2)),
    'min_samples_leaf': scope.int(hp.quniform("min_samples_leaf", 4, 24, 2)),
    'max_samples': scope.float(hp.uniform("max_samples", 0.1, 1)),
}

model_name = "balance_random_forest"

def train_model(params):
    
    X_train_bal, y_train_bal, X_val, y_val = broadcast_data.value
    
    with mlflow.start_run(nested=True):
      model = RandomForestClassifier(**params, random_state=0)
      model.fit(X_train_bal, y_train_bal)
      
      # Get param from fitted model
      fitted_param = model.get_params()
      
      pred = model.predict(X_val)
      pred_prop = model.predict_proba(X_val)[:,1]
      
      bal_acc = balanced_accuracy_score(y_val, pred)
      fpr, tpr, thresholds = roc_curve(y_val, pred_prop)
      roc_auc = auc(fpr, tpr)
      
      rec_scr = recall_score(y_val, pred)
      f1_scr = f1_score(y_val, pred)
      avg_prc = average_precision_score(y_val, pred_prop)

      mlflow.log_metric('val_bal_acc', bal_acc)
      mlflow.log_metric('val_roc_auc', roc_auc)
      mlflow.log_metric('val_rec_scr', rec_scr)
      mlflow.log_metric('val_f1_scr', f1_scr)
      mlflow.log_metric('val_avg_prc', avg_prc)

      loss = -1.0*avg_prc
        
      return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

spark_trials = SparkTrials(parallelism=10)

broadcast_data = sc.broadcast( (X_train_bal, y_train_bal, X_val, y_val) )

# COMMAND ----------

# params = {
#     'n_estimators':100,
#     'max_depth':4,
#     'max_features': 4,
#     'min_samples_leaf': 4,
#     'max_samples': 0.1
# }
# train_model(params)

# COMMAND ----------

with mlflow.start_run(run_name=model_name):
    best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=150,
    trials=spark_trials
    )

# COMMAND ----------

# with mlflow.start_run(run_name=model_name):
#     best_params = fmin(
#         fn=train_model,
#         space=search_space,
#         algo=tpe.suggest,
#         max_evals=150,
#         trials=SparkTrials(),
#     )

# COMMAND ----------

best_params

# COMMAND ----------

best_model = RandomForestClassifier(max_depth = int(best_params["max_depth"]),
                                            max_features = int(best_params["max_features"]),
                                            min_samples_leaf = int(best_params["min_samples_leaf"]),
                                            n_estimators = int(best_params["n_estimators"]),
                                            max_samples = float(best_params["max_samples"]),
                                            n_jobs = -1)
mlflow.start_run(run_name="evaluation", nested=True)
best_model.fit(X_train_bal, y_train_bal)
get_plot_metrics(best_model, X_test, y_test) #
mlflow.end_run()

# COMMAND ----------

importances = best_model.feature_importances_

# COMMAND ----------

feature_names = X_train_bal.columns

# COMMAND ----------

feature_names

# COMMAND ----------

importances

# COMMAND ----------

forest_importances = pd.Series(importances, index=feature_names)
forest_importances = forest_importances.sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.head(10).plot.bar(ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# COMMAND ----------

# shap ** 


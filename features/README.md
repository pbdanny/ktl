# LMP

Features step doccementation

## ground truth

- run main function in features.ground_truth
  - input from feedback staging (abfss path)
  - result will be store in **fdbck_hh_roll.parquet** at feedback storge (abfss path)
- _Example_
  ```python
  from features import ground_truth
  end_date = "2023-02-05"
  test = True # if test is true, it will run on test data
  ground_truth.main(spark, sqlContext, end_date, True)
  ```

## prep transaction data

- run main function in features.transaction
  - input from **cust_details_seg.parquet** from ground truth feedback storge (abfss path)
  - result will be store in **txn_cc_sngl_tndr.parquet** under **_project_name_** storge (abfss path)
- _Example_
  ```python
  from features import transaction
  from edm_class import txnItem # edm class from media shared
  prjct_nm = "test"
  test = True # if test is true, it will run on test data
  transaction.main(spark, dbutils, prjct_nm, txnItem, test=True)
  ```

## features engineering

- run main function in features engineering modules

  - consists of 3 modules
    - **features.quarter_recency**
    - **features.store_format**
    - **features.product_group**
      - details in featuers.json
  - input from **txn_cc_sngl_tndr.parquet** from prep transaction data (abfss path)
  - run using **main** function in each module

- _Example_

  ```python
  from features import quarter_recency
  prjct_nm = "test"
  test = True # if test is true, it will run on test data
  params = [spark, prjct_nm, test]
  quarter_recency.main(*params)
  ```

## combinding features

- run main function in features.combine
  - input from all feature _(features.json)_ from feature engineering (abfss path)
  - result will be store in **all_feature.parquet** under **_project_name_** storge (abfss path)
- _Example_

  ```python
  from features import combine
  prjct_nm = "test"
  test = True # if test is true, it will run on test data
  params = [spark, prjct_nm, test]
  combine.main(*params)
  ```

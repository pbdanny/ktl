# LMP
Customer Lead gereration project
# Documentation
> This project is developing with Azure's Databricks (PySpark)  

[Databrick](https://learn.microsoft.com/en-us/azure/databricks/)\
[PySpark](https://spark.apache.org/docs/latest/api/python/)\
Python package management: [miniconda](https://docs.conda.io/en/latest/miniconda.html)
# Flow
## ETL
### Staging
- run staging file using data appending
  - e.g., __python__ staging.py -- insurer AZAY --lot 202211  --feedback 1
## Model Phase
### ground truth
- run main function in features.ground_truth 
  - input from feedback staging (abfss path)
  - result will be store in __fdbck_hh_roll.parquet__ at feedback storge (abfss path)
  


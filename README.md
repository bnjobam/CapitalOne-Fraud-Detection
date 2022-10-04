# CapitalOne-Fraud-Detection
This is analysis of Caoital ones fraud detection data. The structure is as such: 
exploration.py : to etract the data, drop data points with missing values, determine the distributions of the variables and plot for visual exploration, the data is then stored as capitaldf
wrangling.py : the data is then arranged per account number and more variables are engineered like; running average, number of transactions and transaction rate (using account open date) using transaction amount and transaction time stamp. The final data is atored as fraud_detection
model_prep.py : the data is then separated to train and test partitions. Since the data has more of the no fraud data points we balance them up by appending multiples of isFraud data point to have aproximately the same number of fraud and no frad data points.
4 models were ran: xgmodel.py (extreme gradient boosted trees), abmodel.py (adaptive gradient boosted trees), dcmodel.py (decision trees) and rfmodel.py (random forest with 100 estimators). 
DataScience Challenge.docx is the report of the exercise.
FRAUD DETECTION ACCURACY OF 0.999

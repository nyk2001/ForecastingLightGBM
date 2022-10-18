# Databricks notebook source
# MAGIC %md
# MAGIC ### Install LightGBM Library

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %md
# MAGIC ### Introduction
# MAGIC 
# MAGIC Kernel for the [demand forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only) Kaggle competition.
# MAGIC 
# MAGIC Answer some of the questions posed:
# MAGIC 
# MAGIC * What's the best way to deal with seasonality?
# MAGIC * Should stores be modeled separately, or can you pool them together?
# MAGIC * Does deep learning work better than ARIMA?
# MAGIC * Can either beat xgboost?

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Load Packages

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data

# COMMAND ----------

import os
import pyspark.pandas as ps

# Read pandas dataframe
data_df = pd.read_csv(os.path.join(os.getcwd(),'train.csv'))

# Convert pandas to pandas API
pd_spark_df = ps.DataFrame(data_df)
pd_spark_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Train Data Split

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting train data into train, test, and validation datasets

# COMMAND ----------

def split_data(train_data):
    
    # Break down the date object into month, day, and year
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data['month'] = train_data['date'].dt.month
    train_data['day'] = train_data['date'].dt.dayofweek
    train_data['year'] = train_data['date'].dt.year

    # Include all columns 
    col = [i for i in train_data.columns if i not in ['date','sales']]
    y = 'sales'
    
    # First split to get train and test set
    training_x, val_x, training_y, val_y = train_test_split(train_data[col],train_data[y],stratify = train_data['store'], test_size=0.2, random_state=2018)
    
    # Split train into train and test set
    train_x, test_x, train_y, test_y = train_test_split(training_x[col],training_y, test_size=0.2, random_state=2018,stratify = training_x['store']) 
    
    train_x.reset_index(drop=True, inplace=True)
    test_x.reset_index(drop=True, inplace=True)
    val_x.reset_index(drop=True, inplace=True)
    
    train_y.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)
    val_y.reset_index(drop=True, inplace=True)
    
    return train_x, test_x, val_x, train_y, test_y, val_y

# Split data
train_x, test_x, val_x, train_y, test_y, val_y= split_data(data_df)

# COMMAND ----------

print("** Array Shapes **")
print("Train")
print(train_x.shape)
print(train_y.shape)
print("Test")
print(test_x.shape)
print(test_y.shape)
print("Val")
print(val_x.shape)
print(val_y.shape)

# COMMAND ----------

# Data elements per store
print(train_x.groupby("store").count()["item"])
print(test_x.groupby("store").count()["item"])
print(val_x.groupby("store").count()["item"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model - Simple no tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Model preparation

# COMMAND ----------

def get_model_parameters(search_params):
    fixed_params={
            'nthread': 10,
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression_l1',
            'metric': 'mape', 
           'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 3.097758978478437,
            'lambda_l2': 2.9482537987198496,
            'verbose': 1,
            'min_child_weight': 6.996211413900573,
            'min_split_gain': 0.037310344962162616,
    }
    
    if not bool(search_params): # no tuning
        params = {
                    'nthread': fixed_params['nthread'],
                    'task': fixed_params['task'],
                    'boosting_type': fixed_params['boosting_type'],
                    'objective':  fixed_params['objective'],
                    'metric':  fixed_params['metric'],
                   'bagging_fraction':  fixed_params['bagging_fraction'],
                    'bagging_freq':  fixed_params['bagging_freq'],
                    'lambda_l1':  fixed_params['lambda_l1'],
                    'lambda_l2':  fixed_params['lambda_l2'],
                    'verbose':  fixed_params['verbose'],
                    'min_child_weight':  fixed_params['min_child_weight'],
                    'min_split_gain':  fixed_params['min_split_gain'],
                    'learning_rate' : 0.1,
                    'max_depth': 3, 
                    'num_leaves': 7
                    }
    else:
        params = {
                    'nthread': fixed_params['nthread'],
                    'task': fixed_params['task'],
                    'boosting_type': fixed_params['boosting_type'],
                    'objective':  fixed_params['objective'],
                    'metric':  fixed_params['metric'],
                   'bagging_fraction':  fixed_params['bagging_fraction'],
                    'bagging_freq':  fixed_params['bagging_freq'],
                    'lambda_l1':  fixed_params['lambda_l1'],
                    'lambda_l2':  fixed_params['lambda_l2'],
                    'verbose':  fixed_params['verbose'],
                    'min_child_weight':  fixed_params['min_child_weight'],
                    'min_split_gain':  fixed_params['min_split_gain'],
                    **search_params
                }
    return params

# Train model
def train_model(train_x,train_y,test_x,test_y, search_params):
    
    # Get parameters from models
    parameters = get_model_parameters(search_params)
    
    # Create datasets
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(test_x,test_y)
    
    iterations =3000
    
    # Model crated
    model = lgb.train(parameters, 
                      lgb_train, 
                      iterations, 
                      valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, 
                      verbose_eval=50
                     )
    
    # Return the best map score and the model
    score = model.best_score['valid_1']['mape']
    
    return model, score

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Use MLFlow for model tracking

# COMMAND ----------

import mlflow 

with mlflow.start_run() as run:
    mlflow.lightgbm.autolog()
    model , score = train_model(train_x,train_y,test_x,test_y,{})
    mlflow.log_param("mape", score)

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
logged_model = 'runs:/705580fbeb834b16be1efa0742878055/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(val_x)
rmse = mean_squared_error(val_y.values, predictions)
mape = mean_absolute_percentage_error(val_y.values, predictions)

print("RMSE on val dataset is = %0.3f"%rmse)
print("MAPE on val dataset is = %0.3f"%mape)

# COMMAND ----------



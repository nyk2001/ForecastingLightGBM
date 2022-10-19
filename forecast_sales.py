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
    training_x, test_x, training_y, test_y = train_test_split(train_data[col],train_data[y],stratify = train_data['store'], test_size=0.2, random_state=2018)
    
    # Split train into train and test set
    train_x, val_x, train_y, val_y = train_test_split(training_x[col],training_y, test_size=0.2, random_state=2018,stratify = training_x['store']) 
    
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

# Check for any null values
print(train_x.isnull().values.any())
print(test_x.isnull().values.any())
print(val_x.isnull().values.any())

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
def train_model(train_x,train_y,test_x,test_y,iterations, search_params):
    # Get parameters from models
    parameters = get_model_parameters(search_params)
    
    # Create datasets
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(test_x,test_y)
    
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
experiment_id =-1
run_id =-1

try:
    experiment_id = mlflow.create_experiment("/Users/nabekhan@deloitte.com.au/forecasting-lightgbm")
except Exception as e:
    print(str(e))
    experiment_details = mlflow.get_experiment_by_name("/Users/nabekhan@deloitte.com.au/forecasting-lightgbm")
    experiment_id = experiment_details.experiment_id

with mlflow.start_run(run_name= "LightGBM", experiment_id=experiment_id) as run:
    mlflow.lightgbm.autolog()
    model , score = train_model(train_x,train_y,val_x,val_y,3000, {})
    mlflow.log_param("mape", score)
    print('The best MAPE for validation = %0.3f'%score)
    ax = lgb.plot_importance(model, max_num_features=10)
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    run = mlflow.active_run()
    run_id = run.info.run_id
    
   

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

model_path = f"runs:/{run_id}/model"
    
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_path)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(test_x)
rmse = mean_squared_error(test_y.values, predictions)
mape = mean_absolute_percentage_error(test_y.values, predictions)
print("The best MAPE score for validation dataset = %0.3f"%score)
print("RMSE on test dataset is = %0.3f"%rmse)
print("MAPE on test dataset is = %0.3f"%mape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model - Hyperopt

# COMMAND ----------

from hyperopt import hp, fmin,tpe, Trials

def objective_function(params):
    
    parameters={
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
            'learning_rate' : params['lr'],
            'max_depth': int(params['max_depth']), 
            'num_leaves': int(params['num_leaves'])
    }
    
    # Extract datasets
    lgb_train = params['lgb_train']
    lgb_valid = params['lgb_valid']
    
    iterations =1500
    
    # Model crated
    model = lgb.train(parameters, lgb_train, 
                      iterations, valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, verbose_eval=50)
    
    # Return the best map score and the model
    mape = model.best_score['valid_1']['mape']
    #mlflow.log_param("mape", mape)
    
    ax = lgb.plot_importance(model, max_num_features=10)
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    return mape

# COMMAND ----------

from hyperopt import hp, fmin,tpe, Trials
import mlflow

# Create datasets
lgb_train = lgb.Dataset(train_x,train_y)
lgb_valid = lgb.Dataset(test_x,test_y)

search_space = {
        'lr': hp.uniform('lr', 0.1, 0.5),
        'max_depth': hp.quniform('max_depth', 3,19,1),
        'num_leaves': hp.quniform('num_leaves', 3, 19,1),
        'lgb_train': lgb_train,
        'lgb_valid': lgb_valid
}

experiment_id =-1

num_evals = 15
trials = Trials()
best_hyperparameter = fmin(
        fn=objective_function,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_evals,
        trials=trials,
        rstate=np.random.default_rng(42))
    
try:
    experiment_id = mlflow.create_experiment("/Users/nabekhan@deloitte.com.au/forecasting-lightgbm-hyperopt")
except Exception as e:
    experiment_details = mlflow.get_experiment_by_name("/Users/nabekhan@deloitte.com.au/forecasting-lightgbm-hyperopt")
    experiment_id= experiment_details.experiment_id

print("Best Hyperparameters")
print("Depth = %d"%best_hyperparameter['max_depth'])
print("Learning Rate = %f"%best_hyperparameter['lr'])
print("Number of leaves = %d"%best_hyperparameter['num_leaves'])



# COMMAND ----------

run_id =-1
with mlflow.start_run(experiment_id = experiment_id, 
                      tags={"model": "lightGBM", 
                            "problem": "forecasting"},
                      nested = True):
    
    # Updatinng the best paramters
    seearch_space = {"max_depth":int(best_hyperparameter['max_depth']),
                    "learning_rate":best_hyperparameter['lr'] ,
                    "num_leaves":int(best_hyperparameter['num_leaves'])}
        
    # Train model
    forecast_model , score = train_model(train_x,train_y,val_x,val_y,3000, seearch_space)
    
    # Log model
    mlflow.lightgbm.log_model(forecast_model, "lightgbm-model", input_example=train_x.head(10))
    
    mlflow.log_param("mape", score)
    print('The best MAPE for validation = %0.3f'%score)
    
    ax = lgb.plot_importance(forecast_model, max_num_features=10)
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_hyperparameter['max_depth'])
    mlflow.log_param("numLeaves", best_hyperparameter['num_leaves'])
    mlflow.log_metric("learningRate", best_hyperparameter['lr'])
    
    # Get run ids
    run = mlflow.active_run()
    run_id = run.info.run_id
    print(run_id)

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

model_path = f"runs:/{run_id}/lightgbm-model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_path)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(test_x)
rmse = mean_squared_error(test_y.values, predictions)
mape = mean_absolute_percentage_error(test_y.values, predictions)
print("The best MAPE score for validation dataset = %0.3f"%score)
print("RMSE on test dataset is = %0.3f"%rmse)
print("MAPE on test dataset is = %0.3f"%mape)

# COMMAND ----------



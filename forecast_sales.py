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
data_path = os.path.join(os.getcwd(),'train.csv')
data_df = pd.read_csv(data_path)
data_df['date'] = pd.to_datetime(data_df['date'])
print(type(data_df))
# Convert pandas to pandas API
pd_spark_df = ps.DataFrame(data_df)
print(type(pd_spark_df))
pd_spark_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Train Data Split

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting train data into train, test, and validation datasets

# COMMAND ----------

# Fine for non-time series data
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

import random

def test_train_split_timeseries_data(data_df, split_ratio, random_ratio= False):
    # Break down the date object into month, day, and year
    pd_spark_df['month'] = pd_spark_df['date'].dt.month
    pd_spark_df['day'] = pd_spark_df['date'].dt.dayofweek
    pd_spark_df['year'] = pd_spark_df['date'].dt.year

    train_split = split_ratio
    
    # Find the unique store ids
    unique_elements = pd_spark_df['store'].unique().to_numpy()
    unique_elements= np.sort(unique_elements,axis=0)
    
    # Creating new train and test data sets
    df_train = ps.DataFrame()
    df_test = ps.DataFrame()
    
    for each_element in unique_elements:
        temp_df = {}
        
        # Extract sales data for each store
        temp_spark_df = data_df[data_df['store']==int(each_element)].copy()
        
        # Sort data with respect to the date column
        temp_spark_df = temp_spark_df.sort_values(by=['date'], ascending=True)
        temp_spark_df.reset_index(drop=True, inplace=True)
        
        if random_ratio:
            train_split = random.uniform(0.75, 0.85)
        
        total_elements = temp_spark_df.shape[0]
        train_elements = int(temp_spark_df.shape[0]*train_split)
        
        # First 90% for training
        df_train= df_train.append(temp_spark_df[0:train_elements])
        
        # Remaining 10% for testing
        df_test=  df_test.append(temp_spark_df[train_elements:])

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Train columns
    col = [i for i in df_train.columns if i not in ['date','sales']]
    y = 'sales'

    return df_train[col],df_train[y], df_test[col], df_test[y],df_train, df_test

temp_x, temp_y, test_x, test_y, df_train, df_test = test_train_split_timeseries_data(pd_spark_df,0.9)
train_x, train_y, val_x, val_y, df_train, df_test = test_train_split_timeseries_data(df_train, 0.8, True)

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
    
    # Create datasets (converting pyspark to numpy)
    lgb_train = lgb.Dataset(train_x.to_numpy(), train_y.to_numpy())
    lgb_valid = lgb.Dataset(test_x.to_numpy(), test_y.to_numpy())
    
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
#The Model signature defines the schema of a model's inputs and outputs. Model inputs and outputs can be either column-based or tensor-based.
from mlflow.models.signature import infer_signature
from mlflow.tracking.client import MlflowClient

experiment_id =-1
run_id =-1

try:
    experiment_id = mlflow.create_experiment("/Users/nabekhan@deloitte.com.au/forecasting-lightgbm")
except Exception as e:
    experiment_details = mlflow.get_experiment_by_name("/Users/nabekhan@deloitte.com.au/forecasting-lightgbm")
    experiment_id = experiment_details.experiment_id

project_description = ("Train LightGBM model without any parameter tuning. The trained model may not perform very well on test data as it is not optimized")
with mlflow.start_run(run_name= "LightGBM", 
                      tags={"Problem": "Timeseries forecasting", 
                             "mlflow.note.content": project_description}, 
                      experiment_id=experiment_id,
                      nested=True) as run:
    mlflow.lightgbm.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
    model , score = train_model(train_x,train_y,val_x,val_y,3000, {})
    
    print('The best MAPE for validation = %0.3f'%score)
    ax = lgb.plot_importance(model, max_num_features=10)
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    signature = infer_signature(train_x.to_numpy(), model.predict(train_x.to_numpy()))
    
    mlflow.log_metric("mape", score)
    mlflow.log_param("model", "lightGBM")
    mlflow.log_artifact(data_path, artifact_path="train_data")
    
# Getting the current run id
run_id = run.info.run_id

# Register the model
model_uri = f"runs:/{run_id}/model"
model_name= 'lightGBM_simple_model'
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# Update model details
client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)
print(model_version_details)

client.update_model_version(name=model_details.name,version=model_details.version,
    description="This model predicts time series forecasting.")

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

model_path = f"runs:/{run_id}/model"
    
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_path)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(test_x.to_numpy())
rmse = mean_squared_error(test_y.values, predictions)
mape = mean_absolute_percentage_error(test_y.values, predictions)
print("The best MAPE score for validation dataset = %0.3f"%score)
print("RMSE on test dataset is = %0.3f"%rmse)
print("MAPE on test dataset is = %0.3f"%mape)

# Computing RMSE and MAPE for each individual store
test_df = ps.concat([test_y, test_x], axis=1)
stores = test_df['store'].unique().to_numpy()
stores = np.sort(stores,axis=0)

metric_outputs= list()

for each_store_id in stores:
    # Extract test dataset for each store
    subset_df = test_df[test_df['store']==int(each_store_id)].copy()
    subset_df.reset_index(drop=True, inplace=True)
    subset_test_x = subset_df.loc[:, subset_df.columns != 'sales']
    subset_test_y = subset_df['sales']
    
    # Predict on a Pandas DataFrame
    predictions = loaded_model.predict(subset_test_x.to_numpy())
    rmse = mean_squared_error(subset_test_y.values, predictions)
    mape = mean_absolute_percentage_error(subset_test_y.values, predictions)
    
    metric_outputs.append([each_store_id, rmse, mape])

results_df = pd.DataFrame(metric_outputs,columns=['Store Id','RMSE','MAPE'])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(results_df['MAPE'])
plt.axhline(y=np.nanmean(results_df['MAPE']),linewidth=2, color='r')
plt.xlabel("Stores")
plt.ylabel("MMAPE")
plt.title("MAPE for each store")
plt.subplot(1,2,2)
plt.xlabel("Stores")
plt.ylabel("RMSE")
plt.plot(results_df['RMSE'])
plt.axhline(y=np.nanmean(results_df['RMSE']),linewidth=2, color='r')
plt.title("RMSE for each store")

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
            'bagging_freq': 5,
            'lambda_l1': 3.097758978478437,
            'lambda_l2': 2.9482537987198496,
            'verbose': 1,
            'min_child_weight': 6.996211413900573,
            'min_split_gain': 0.037310344962162616,
            'learning_rate' : params['lr'],
            'max_depth': int(params['max_depth']), 
            'num_leaves': int(params['num_leaves']),
            'bagging_fraction': params['bagging_fraction'],
            'feature_fraction' : params['feature_fraction']
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
lgb_train = lgb.Dataset(train_x.to_numpy(),train_y.to_numpy())
lgb_valid = lgb.Dataset(test_x.to_numpy(),test_y.to_numpy())

search_space = {
        'lr': hp.uniform('lr', 0.1, 0.5),
        'max_depth': hp.quniform('max_depth', 3,19,1),
        'num_leaves': hp.quniform('num_leaves', 3, 19,1),
        'lgb_train': lgb_train,
        'lgb_valid': lgb_valid,
        'bagging_fraction':  hp.uniform('bagging_fraction', 0.8,1.0),
        'feature_fraction' : hp.uniform('feature_fraction', 0.8,1.0)
}

experiment_id =-1

num_evals = 30
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
print("Bagging fraction = %d"%best_hyperparameter['bagging_fraction'])
print("Feature fraction = %d"%best_hyperparameter['feature_fraction'])



# COMMAND ----------

print("Best Hyperparameters")
print("Depth = %d"%best_hyperparameter['max_depth'])
print("Learning Rate = %f"%best_hyperparameter['lr'])
print("Number of leaves = %d"%best_hyperparameter['num_leaves'])
print("Bagging fraction = %f"%best_hyperparameter['bagging_fraction'])
print("Feature fraction = %f"%best_hyperparameter['feature_fraction'])

run_id =-1
with mlflow.start_run(experiment_id = experiment_id, 
                      tags={"model": "lightGBM", 
                            "problem": "forecasting"},
                      nested = True):
    
    # Updatinng the best paramters
    seearch_space = {"max_depth":int(best_hyperparameter['max_depth']),
                    "learning_rate":best_hyperparameter['lr'],
                    "num_leaves":int(best_hyperparameter['num_leaves']),
                    "bagging_fraction":best_hyperparameter['bagging_fraction'],
                    "feature_fraction": best_hyperparameter['feature_fraction']}
        
    # Train model
    forecast_model , score = train_model(train_x,train_y,val_x,val_y,3000, seearch_space)
    
    # Log model
    mlflow.lightgbm.log_model(forecast_model, "lightgbm-model", input_example=train_x.head(5).to_numpy())
    
    mlflow.log_param("mape", score)
    print('The best MAPE for validation = %0.3f'%score)
    
    ax = lgb.plot_importance(forecast_model, max_num_features=10)
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    # Log param and metrics for the final model
    mlflow.log_metric("maxDepth", best_hyperparameter['max_depth'])
    mlflow.log_metric("numLeaves", best_hyperparameter['num_leaves'])
    mlflow.log_metric("learningRate", best_hyperparameter['lr'])
    mlflow.log_metric("bagging_fraction", best_hyperparameter['bagging_fraction'])
    mlflow.log_metric("feature_fraction", best_hyperparameter['feature_fraction'])
    
    # Get run ids
    run = mlflow.active_run()
    run_id = run.info.run_id
    

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

model_path = f"runs:/{run_id}/lightgbm-model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_path)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(test_x.to_numpy())
rmse = mean_squared_error(test_y.values, predictions)
mape = mean_absolute_percentage_error(test_y.values, predictions)
print("The best MAPE score for validation dataset = %0.3f"%score)
print("RMSE on test dataset is = %0.3f"%rmse)
print("MAPE on test dataset is = %0.3f"%mape)

# Computing RMSE and MAPE for each individual store
test_df = ps.concat([test_y, test_x], axis=1)
stores = test_df['store'].unique().to_numpy()
stores = np.sort(stores,axis=0)

metric_outputs= list()

for each_store_id in stores:
    # Extract test dataset for each store
    subset_df = test_df[test_df['store']==int(each_store_id)].copy()
    subset_df.reset_index(drop=True, inplace=True)
    subset_test_x = subset_df.loc[:, subset_df.columns != 'sales']
    subset_test_y = subset_df['sales']
    
    # Predict on a Pandas DataFrame
    predictions = loaded_model.predict(subset_test_x.to_numpy())
    rmse = mean_squared_error(subset_test_y.values, predictions)
    mape = mean_absolute_percentage_error(subset_test_y.values, predictions)
    
    metric_outputs.append([each_store_id, rmse, mape])

results_df = pd.DataFrame(metric_outputs,columns=['Store Id','RMSE','MAPE'])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(results_df['MAPE'])
plt.axhline(y=np.nanmean(results_df['MAPE']),linewidth=2, color='r')
plt.xlabel("Stores")
plt.ylabel("MMAPE")
plt.title("MAPE for each store")

plt.subplot(1,2,2)
plt.xlabel("Stores")
plt.ylabel("RMSE")
plt.plot(results_df['RMSE'])
plt.axhline(y=np.nanmean(results_df['RMSE']),linewidth=2, color='r')
plt.title("RMSE for each store")

# COMMAND ----------



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
# MAGIC ### Load data in pandas

# COMMAND ----------

import os

# Read csv data into pandas dataframe
data_path = os.path.join(os.getcwd(),'train.csv')
data_df = pd.read_csv(data_path)
print("Total elements = %d"%data_df.shape[0])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# Generating extra features such as week of year, day of week etc
from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)

def create_date_features(df):
    data_df['date'] = pd.to_datetime(data_df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year    
    df['week_of_year'] = df['date'].dt.weekofyear # Which week of the corresponding year
    df['day_of_week'] = df['date'].dt.dayofweek # Which day of the corresponding week of the each month
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int) # Is it starting of the corresponding month
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int) # Is it ending of the corresponding month
    
    return df


data_df = create_date_features(data_df)
data_df.head()


# COMMAND ----------

print("Total elements = %d"%data_df.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Lagged features

# COMMAND ----------

# MAGIC %md
# MAGIC *Adding "Random Noise" to the "Lag/Shifted Features"*
# MAGIC * We are generating these "Lag/Shifted Features" from the target variable "sales", actually we are causing a problem which is called **data leakage** in data science literature. 
# MAGIC * The reason of that **data leakage** problem is that in our case, normally we shouldn't generate features by using target variable when we are working on ML project. Because it causes **overfitting** to the train data. Model notices target variable base features explains "target" column well, and focuses more to that columns. Consequently, it loses its "generalization" ability.
# MAGIC * This will cause, model will not to learn the exact values of target variable and as a result we avoid "overfitting" situation

# COMMAND ----------

# Let's first create "Random Noise" function, for using when we want 
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),)) # Gaussian random noise


# And let's create "Lag/Shifted Features" by using this function
# Since we will create more than 1 "Lag/Shifted Features" I created that function.
def create_lagged_features(df, lags):
    for lag in lags:
        df['sales_lag_' + str(lag)] = df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag))+ random_noise(df)
    return df


#def create_lagged_features(df):    
#    # Let's see an example : With this code we generate 'lag1' feature for ecah unique 'item' in each 'store' separately.
#    df['lag_1']= df.groupby(["store", "item"])[['sales']].transform(lambda x: x.shift(1))
#    df['lag_2']= df.groupby(["store", "item"])[['sales']].transform(lambda x: x.shift(2))
#    df['lag_3']= df.groupby(["store", "item"])[['sales']].transform(lambda x: x.shift(3))
#    
#    return df
print("Total elements = %d"%data_df.shape[0])
lagged_data_df = create_lagged_features(data_df,[1,3,5,7])
print("Total elements = %d"%lagged_data_df.shape[0])
lagged_data_df.head(10)   

# COMMAND ----------

# MAGIC %md
# MAGIC #### Rolling Window

# COMMAND ----------

# MAGIC %md
# MAGIC * "Moving Average Method" is used for forcasting "Time Series" problems. This method simply takes "n" previous target variable and averages them and returns as a new value.
# MAGIC * So since we know that, this kind of method is used for forcasting "Time Series" problems, again we generate new feature by using that method.
# MAGIC * So since we said that while using ML approach we have to generate features that represent time series patterns, we actually get help from traditional methods for that purpose.

# COMMAND ----------

# Let's create "rolling mean features".
def roll_mean_features(df, windows):
    for window in windows:
        df['sales_roll_mean_' + str(window)] = df.groupby(["store", "item"])['sales'].\
        transform(lambda x: x.shift(1).rolling(window=window, min_periods=int(window/2)).mean()) + random_noise(df)
    return df

# Let's create "rolling sum features".
def roll_sum_features(df, windows):
    for window in windows:
        df['sales_roll_sum_' + str(window)] = df.groupby(["store", "item"])['sales'].\
        transform(lambda x: x.shift(1).rolling(window=window, min_periods=int(window/2)).sum()) + random_noise(df)
    return df

# Using two window sizes
windows_size = [3,7] 
rolled_data_df = roll_mean_features(lagged_data_df, windows_size)
rolled_data_df = roll_sum_features(rolled_data_df, windows_size)
print("Total elements = %d"%rolled_data_df.shape[0])
rolled_data_df.head(10)  


# COMMAND ----------

print("Total elements = %d"%rolled_data_df.shape[0])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exponentially Weighted Mean Features
# MAGIC 
# MAGIC * Another traditional "Time Series Method" is "Exponentially Weighted Mean" method. This method has parameter called _alpha_ used as smoothing factor. This parameter ranges between [0, 1]. If _alpha_ is close to 1 while taking average for last for instance 10 days(rolling mean features also was taking averages but without giving weight), it gives more _weight_ to the close days and decreases the _weight_ when going to more past days.  
# MAGIC * You can read about this method more on internet, but briefly normally in time series forecatsing it's better to give more _weights_ to the more recent days rather tham giving same _weight_ to all past days.
# MAGIC * Because more recent days have more influence to the current day. Therefore, giving more _weight_ to the more recent days makes sense.
# MAGIC * This method uses that formula behind in its calculations(xt : past days values) : 
# MAGIC 
# MAGIC * As we see when it goes more past values it decreases the _weight_

# COMMAND ----------

# Let's create 'Exponentially Weighted Mean Features' 
def ewm_features(df, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            df['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    
    return df

# In here we have two combinations : alphas and lags. Agian we give variety of variables for both and will se which one is best. 
alphas = [1.0, 0.95]
lags = [3,7]

final_data_df = ewm_features(rolled_data_df, alphas, lags)
final_data_df.head(10)

# COMMAND ----------

print("Total elements = %d"%final_data_df.shape[0])


# COMMAND ----------

import pyspark.pandas as ps

print("Total elements = %d"%final_data_df.shape[0])
final_data_df.dropna(inplace=True)
print("Total elements after removing nulls = %d"%final_data_df.shape[0])

# Convert pandas to pandas API
pd_spark_df = ps.DataFrame(final_data_df)
pd_spark_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Train Data Split

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting train data into train, test, and validation datasets

# COMMAND ----------

import random

def test_train_split_timeseries_data(data_df, split_ratio, random_ratio= False):
    
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

# Use 80% for training and 20% for testing
temp_x, temp_y, test_x, test_y, df_train, df_test = test_train_split_timeseries_data(pd_spark_df,0.8)
# Use 80%for training and 20% for validation
train_x, train_y, val_x, val_y, df_train, df_test = test_train_split_timeseries_data(df_train, 0.8)

# COMMAND ----------

train_x.head()

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

store_metric_outputs= list()
item_metric_outputs = list()
for each_store_id in stores:
    # Extract test dataset for each store
    store_df = test_df[test_df['store']==int(each_store_id)].copy()
    store_df.reset_index(drop=True, inplace=True)
    store_test_x = store_df.loc[:, store_df.columns != 'sales']
    store_test_y = store_df['sales']
    
    # Predict on a Pandas DataFrame - at store level
    predictions = loaded_model.predict(store_test_x.to_numpy())
    rmse = mean_squared_error(store_test_y.values, predictions)
    mape = mean_absolute_percentage_error(store_test_y.values, predictions)
    
    store_metric_outputs.append([each_store_id, rmse, mape])

store_results_df = pd.DataFrame(store_metric_outputs,columns=['StoreId','RMSE','MAPE'])
#item_results_df = pd.DataFrame(item_metric_outputs,columns=['StoreId','ItemId','RMSE','MAPE'])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(store_results_df['MAPE'])
plt.axhline(y=np.nanmean(store_results_df['MAPE']),linewidth=2, color='r')
plt.xlabel("Stores")
plt.ylabel("MMAPE")
plt.title("MAPE for each store")
plt.subplot(1,2,2)
plt.xlabel("Stores")
plt.ylabel("RMSE")
plt.plot(store_results_df['RMSE'])
plt.axhline(y=np.nanmean(store_results_df['RMSE']),linewidth=2, color='r')
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



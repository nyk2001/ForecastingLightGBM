# Databricks notebook source
# MAGIC %md
# MAGIC # Install LightGBM Library

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction
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
# MAGIC # Load Packages

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load data in PySpark and Pandas Dataframe API

# COMMAND ----------

import os
import pyspark.pandas as ps

# Assuming we already have a table sales_train_data in default Hive database
spark_df=spark.sql("select * from sales_train_data")

# Converting table to pandas API data frame - comparison
#pandas_df_api= ps.DataFrame(spark_df)
pandas_df_api= spark_df.toPandas()

print("Total elements = %d"%pandas_df_api.shape[0])
print("Type of frame = %s"%type(pandas_df_api))

# COMMAND ----------

spark_df.limit(5).show()

# COMMAND ----------

pandas_df_api.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date fetaures from date

# COMMAND ----------

# MAGIC %md
# MAGIC Using Pandas Datafram API

# COMMAND ----------

# Generating extra features such as week of year, day of week etc
from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)

# Creates date features using pandas data frame api
# Input
#   --> Pandas API Dataframe
# Output
#   Dataframe with various time outputs
def create_date_features_pandas(df):
    df['date'] = ps.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year    
    df['week_of_year'] = df['date'].dt.weekofyear # Which week of the corresponding year
    df['day_of_week'] = df['date'].dt.dayofweek # Which day of the corresponding week of the each month
    df["is_weekend"] = (df.date.dt.weekday // 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int) # Is it starting of the corresponding month
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int) # Is it ending of the corresponding month
    
    return df

# Generate time features based on the date column
pandas_df_api_v1 = create_date_features_pandas(pandas_df_api)
pandas_df_api_v1.head()


# COMMAND ----------

# MAGIC %md
# MAGIC Using PySpark

# COMMAND ----------

from pyspark.sql.types import DateType, FloatType, IntegerType
from pyspark.sql.functions import year, month, dayofmonth, to_date, col, weekofyear, dayofweek,last_day,to_timestamp

# Creates date features using pandas data frame api
# Creates date features using pandas data frame api
# Input
#   --> Pandas API Dataframe
# Output
#   Dataframe with various time outputs
def create_date_features_pyspark(df):
    df = df.withColumn("date", to_date(col("date"),"MM-dd-yyyy")) 
    df = df.withColumn("timestamp", to_timestamp(col("date"))) 
    df = df.withColumn("month",month(df['date']))
    df = df.withColumn("day",dayofmonth(df['date']))
    df = df.withColumn("year",year(df['date']))
    df = df.withColumn("week_of_year",weekofyear(df['date']))
    df = df.withColumn("day_of_week",((dayofweek(df['date'])+5)%7))
    df = df.withColumn("is_weekend", dayofweek(df['date']).isin([7,1]).cast(IntegerType()))
    df = df.withColumn("is_month_start", (dayofmonth(df['date'])==1).cast(IntegerType()))
    df = df.withColumn("is_month_end", ((dayofmonth(last_day(df['date'])))==dayofmonth(df['date'])).cast(IntegerType()))
    
    return df

# Create pyspark data frame 
spark_df_v1= create_date_features_pyspark(spark_df)
spark_df_v1.limit(10).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Create subset of data

# COMMAND ----------

#df1 =spark_df_v1.filter(((col("store")==1) & (col("item")==1) & (col("month")==1) & (col("day")<=15) & (col("year")==2013)) | ((col("store")==1) & (col("item")==2) & (col("month")==1) & (col("day")<=15) & #(col("year")==2013)))
#spark_df_v1 = df1
#pandas_df_api_v1 = spark_df_v1.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Misc Functions

# COMMAND ----------

from pyspark.sql import DataFrame
##############################################################################
# Save table to a default schema 
##############################################################################
# Input
#  --> Dataframe. Can be pandas or spark
#  --> Table name
# Output
#  --> None
def write_data_hive(df, table_name):
    # Convert from pandas API data frame to spark data frame
    if not isinstance(df, DataFrame):
        df = df.to_spark()
    
    # Write on disc
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)

##############################################################################
# Compare column values between pandas and PySpark data frame
##############################################################################
# Input
#  --> Pyspark dataframe
#  --> Pandas dataframe
#  --> Group cols, such as ['store', 'item','date']
#  --> List of columns to be compared
# Output
#  --> None
def comapre_data(spark_df, pandas_df, group_cols, col_names):
    # Converting all data frames to Pandas API
    df1 = ps.DataFrame(spark_df)
    df2 = ps.DataFrame(pandas_df)
    
    # Sort by columns
    df1= df1.sort_values(by=group_cols)
    df2= df2.sort_values(by=group_cols)
    
    # Reset indices
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    for col_name in col_names:
        
        # Fill all null values with a dummy value
        df1[col_name] = df1[col_name].fillna(-1000)
        df2[col_name] = df2[col_name].fillna(-1000)
        
        # Select a subset of columns
        df3 = df1[['date','store', 'item','sales',col_name]].copy()
        df4 = df2[['date','store', 'item',col_name]].copy()
        
        # Merging two data frames
        df_joined= df3.merge(df4, on = ['date', 'store','item'] , how="inner")
        df_joined['diff'] = df_joined['%s_x'%col_name]- df_joined['%s_y'%col_name]
        print(f"{col_name}: Spark dataframe rows ={df2.shape[0]}  PySpark dataframe rows = {df4.shape[0]}") 
        print(f"Joined Dataframe rows = {df_joined.shape[0]}  Total Mismatches = {df_joined[df_joined['diff']>0.001].shape[0]}")

# COMMAND ----------

## Writing data as a hive table
#table_name = 'sales_data'
#write_data_hive(spark_df_v1, 'sale_data_v1')
#write_data_hive(pandas_df_api_v1, 'sale_data_v1')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lagged features

# COMMAND ----------

# MAGIC %md
# MAGIC *Adding "Random Noise" to the "Lag/Shifted Features"*
# MAGIC * We are generating these "Lag/Shifted Features" from the target variable "sales", actually we are causing a problem which is called **data leakage** in data science literature. 
# MAGIC * The reason of that **data leakage** problem is that in our case, normally we shouldn't generate features by using target variable when we are working on ML project. Because it causes **overfitting** to the train data. Model notices target variable base features explains "target" column well, and focuses more to that columns. Consequently, it loses its "generalization" ability.
# MAGIC * This will cause, model will not to learn the exact values of target variable and as a result we avoid "overfitting" situation

# COMMAND ----------

# MAGIC %md
# MAGIC Using Pandas  - Pandas Dataframe API will be very slow

# COMMAND ----------

from pyspark.sql.functions import lit

# And let's create "Lag/Shifted Features" by using this function
# Creating lagged features
# Inputs:
#    --> Pandas Data frame 
#    --> Lag values, such as 1 , 3 , 5
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> lag column suffix, such as lag
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pandas dataframe with new columns
def create_lagged_features_pandas(df, lags,group_cols,  target_col_name, lag_suffix, random_noise= False):
    # Creating col for each lag
    for lag in lags:
        df[target_col_name + "_"+ lag_suffix+ "_" + str(lag)] = df.groupby(group_cols)[target_col_name].transform(lambda x: x.shift(lag)) 
    
    return df

# Lagged windows 
lag_window= [1,3,5,7]

# Generate lagged features
pandas_df_api_v2 = create_lagged_features_pandas(pandas_df_api_v1,lag_window,["store", "item"],"sales", "lag")
print("Total elements = %d"%pandas_df_api_v2.shape[0])
pandas_df_api_v2 = pandas_df_api_v2.sort_values(by=["store", "item","date"], ascending=True)
pandas_df_api_v2.head(10)   

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import numpy as np

# Creating lagged features
# Inputs:
#    --> Pyspark Data frame 
#    --> Lag values, such as 1 , 3 , 5
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> lag column suffix, such as lag
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pyspark dataframe with new columns
def create_lagged_features_pyspark(df,lags,group_cols, target_col_name, lag_suffix, random_noise=False):
    
    # Defining the window to be used to generate lagged features
    w = Window().partitionBy(['store','item']).orderBy(group_cols)

    # Add ranom noise (if needed) - Disabled at the moment
    random_df = np.random.normal(scale=1.6, size=(df.count(),))    
    
    # Generating lags
    for lag in lags:
        df = df.withColumn(f"{target_col_name}_{lag_suffix}_{lag}", F.lag(F.col(target_col_name), lag).over(w)) 
        
    return df

# Lag windows
lag_window= [1,3,5,7]

# Generating lagged features pyspark
spark_df_v2 = create_lagged_features_pyspark(spark_df_v1, lag_window,['store','item'],"sales","lag")
spark_df_v2= spark_df_v2.orderBy("store","item","date")
spark_df_v2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing columns between Pandas and PySpark dataframe

# COMMAND ----------

# Comapring columns
col_list=['sales_lag_1','sales_lag_3','sales_lag_5','sales_lag_7']
comapre_data(spark_df_v2,pandas_df_api_v2,['store', 'item','date'], col_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rolling Window

# COMMAND ----------

# MAGIC %md
# MAGIC * "Moving Average Method" is used for forcasting "Time Series" problems. This method simply takes "n" previous target variable and averages them and returns as a new value.
# MAGIC * So since we know that, this kind of method is used for forcasting "Time Series" problems, again we generate new feature by using that method.
# MAGIC * So since we said that while using ML approach we have to generate features that represent time series patterns, we actually get help from traditional methods for that purpose.

# COMMAND ----------

# MAGIC %md
# MAGIC Pandas Dataframe- Pandas API very slow

# COMMAND ----------

# Let's create "rolling mean features"
# Creating rolling window features
# Inputs:
#    --> Pandas Data frame 
#    --> Windows values, such as 3 , 5 , 7
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Roll column suffix, such as lag
#    --> Operation can be sum or mean
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pandas dataframe with new columns
def create_window_rolling_features(df, window_sizes,group_cols, target_col_name,rolling_suffix, operation, random_noise= False):
    
    for window_size in window_sizes:
        if operation == "sum":
            df["sales" + "_" +rolling_suffix+"_" + operation + "_" + str(window_size)] = df.groupby(group_cols)[target_col_name].\
                transform(lambda x: x.rolling(window=window_size, center= True,min_periods=1).sum())
        if operation =="mean":
            df["sales" + "_" + rolling_suffix+"_"+ operation + "_" + str(window_size)] = df.groupby(group_cols)[target_col_name].\
                transform(lambda x: x.rolling(window=window_size,center= True,min_periods=1).mean())
        if operation =="std":
            df["sales" + "_" + rolling_suffix+"_"+ operation + "_" + str(window_size)] = df.groupby(group_cols)[target_col_name].\
                transform(lambda x: x.rolling(window=window_size,center= True,min_periods=1).std())
        
    return df

# Using two window sizes
windows_sizes = [3,5,7] 
pandas_df_api_v2 = pandas_df_api_v2.sort_values(by= ["store","item","date"], ascending=True)
pandas_df_api_v3 = create_window_rolling_features(pandas_df_api_v2, windows_sizes,["store", "item"], "sales", "roll", "sum")
pandas_df_api_v3 = create_window_rolling_features(pandas_df_api_v3, windows_sizes,["store", "item"], "sales", "roll", "mean")
pandas_df_api_v3 = create_window_rolling_features(pandas_df_api_v3, windows_sizes,["store", "item"], "sales", "roll", "std")
pandas_df_api_v3.head(10)  

# COMMAND ----------

import pyspark.pandas as ps
from pyspark.sql import *
ps.set_option('compute.ops_on_diff_frames', True)

# Let's create "rolling mean features"
# Inputs:
#    --> Pyspark Data frame 
#    --> Windows values, such as 3 , 5 , 7
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Roll column suffix, such as lag
#    --> Operation can be sum or mean
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pyspark dataframe with new columns

def create_rolling_features(df, window_sizes,group_cols, target_col_name,rolling_suffix, operation, random_noise= False):
    
    # Calculating days
    days = lambda i: i * 86400
    
    for window_size in window_sizes:        
        # Computing window delta
        window_delta = int(window_size/2)
        
        # Creating window for roll feature
        roll_window = Window().partitionBy(group_cols).orderBy(F.col("timestamp").cast('long')).rangeBetween(-days(window_delta), days(window_delta))    
        
        # Generate new column
        if operation == "sum":
            df = df.withColumn(f"{target_col_name}_{rolling_suffix}_{operation}_{window_size}", F.sum(target_col_name).over(roll_window))
        if operation == "mean":
            df = df.withColumn(f"{target_col_name}_{rolling_suffix}_{operation}_{window_size}", F.avg(target_col_name).over(roll_window))
        if operation == "std":
            df = df.withColumn(f"{target_col_name}_{rolling_suffix}_{operation}_{window_size}", F.stddev(target_col_name).over(roll_window))
            
    return df

    
# Using two window sizes
windows_sizes = [3,5,7] 

# Generate mean and sum rolling features
spark_df_v2 = spark_df_v2.orderBy("store","item","date")
spark_df_v3 = create_rolling_features(spark_df_v2, windows_sizes,["store","item"], "sales", "roll", "sum")
spark_df_v3 = create_rolling_features(spark_df_v3, windows_sizes,["store","item"], "sales", "roll", "mean")
#spark_df_v3 = spark_df_v3.orderBy("store","item","date")
spark_df_v3 = create_rolling_features(spark_df_v3, windows_sizes,["store","item"], "sales", "roll", "std")
spark_df_v3 = spark_df_v3.orderBy("store","item","date")
spark_df_v3.limit(20).toPandas().head(20)  


# COMMAND ----------

# MAGIC %md
# MAGIC Comapring columns between Pandas Dataframe and PySpark

# COMMAND ----------

# Comapring columns
col_list=['sales_roll_mean_3','sales_roll_mean_5','sales_roll_mean_7','sales_roll_sum_3','sales_roll_sum_5','sales_roll_sum_7','sales_roll_std_3','sales_roll_std_5','sales_roll_std_7']
comapre_data(spark_df_v3,pandas_df_api_v3,['store', 'item','date'], col_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exponentially Weighted Mean Features
# MAGIC 
# MAGIC * Another traditional "Time Series Method" is "Exponentially Weighted Mean" method. This method has parameter called _alpha_ used as smoothing factor. This parameter ranges between [0, 1]. If _alpha_ is close to 1 while taking average for last for instance 10 days(rolling mean features also was taking averages but without giving weight), it gives more _weight_ to the close days and decreases the _weight_ when going to more past days.  
# MAGIC * You can read about this method more on internet, but briefly normally in time series forecatsing it's better to give more _weights_ to the more recent days rather tham giving same _weight_ to all past days.
# MAGIC * Because more recent days have more influence to the current day. Therefore, giving more _weight_ to the more recent days makes sense.
# MAGIC * This method uses that formula behind in its calculations(xt : past days values) : 
# MAGIC 
# MAGIC * As we see when it goes more past values it decreases the _weight_

# COMMAND ----------

from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)

# Let's create 'Exponentially Weighted Mean Features' 
# Inputs:
#    --> Pandas Data frame 
#    --> Alpha values, such as 1 , 0.95
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Ewm column suffix, such as lag
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pandas dataframe with new columns
def create_ewm_features(df, alphas,group_cols, target_col_name,ewm_suffix, random_noise= False):
    
    # For each alpha calculate values
    for alpha in alphas:
        col_name = target_col_name + "_" + ewm_suffix  + "_" + str(alpha).replace('.','_')
        df[col_name] = \
                df.groupby(group_cols)[target_col_name].transform(lambda x: x.ewm(alpha=alpha).mean())
        
    return df

# In here we have two combinations : alphas and lags
alphas = [0.75, 0.95, 1.0]

pandas_df_api_v4 = create_ewm_features(pandas_df_api_v3, alphas, ["store","item"], "sales", "ewm")
pandas_df_api_v4 = pandas_df_api_v4.sort_values(["store","item","date"])
pandas_df_api_v4.head(20) 

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import DoubleType, StructField

# Inputs:
#    --> Pyspark Data frame 
#    --> Alpha values, such as 1 , 0.95
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Ewm column suffix, such as lag
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pyspark dataframe with new columns

def create_ewm_features(df, alphas,group_cols, target_col_name,ewm_suffix, random_noise= False):
    
    # For each alpha calculate values
    for alpha in alphas:
        col_name = target_col_name + "_" + ewm_suffix  + "_" + str(alpha).replace('.','_')
        df = df.withColumn(str(col_name), F.lit(None).cast('double'))
        
        @pandas_udf(df.schema, PandasUDFType.GROUPED_MAP)
        # Using Pyhton function to compute stats for each group
        def ema(pdf):
            col_name = target_col_name + "_" + ewm_suffix + "_" + str(alpha).replace('.','_')
            pdf[str(col_name)] = pdf[target_col_name].ewm(alpha=alpha).mean()
            return pdf
        
        # Split and apply 
        df= df.groupby(group_cols).apply(ema)
    
    return df

alphas= [0.75, 0.95,1.0]    
spark_df_v4=create_ewm_features(spark_df_v3, alphas, ["store","item"], "sales", "ewm")
spark_df_v4.show()

# COMMAND ----------

# Comapring columns
col_list=['sales_ewm_0_75','sales_ewm_0_95','sales_ewm_1_0']
comapre_data(spark_df_v4,pandas_df_api_v4,['store', 'item','date'], col_list)

# COMMAND ----------

#####################################################
#                                                   #
# Data Leakge - Split trina and test data first     #
#                                                   #
#####################################################

#  Seperate module - Featurization module
#           Read data from Delta Lake - Parquet File
#           Modify data using our code.. lagged, rolliong
#           Write data back to Delta Lake - Parquet File
# 
#  Use PyTest to test functions (AWS Cloud 9)

#
# PandasUDF
# Deepchecks 
# Forecast helper house - Predicted 
# Recursive for prediction - makes sense
#
######################################
# Read that document shared by Pal
######################################

# Generating extra features such as week of year, day of week etc
from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)

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
final_data_df = final_data_df.sort_index(ascending=True)
final_data_df.head(10)

# COMMAND ----------

write_data_hive(final_data_df, "sale_train_final_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Train Data Split

# COMMAND ----------

import os
import pyspark.pandas as ps

# Assuming we already have a table sales_train_data in defaule Hive database
spark_df=spark.sql("select * from sale_train_final_data")
#spark_df= spark_df.coalesce(1)
#print(spark_df.rdd.getNumPartitions())
#ps.set_option("compute.default_index_type", "sequence")
# Converting pyspark to pandas API
final_data_df= ps.DataFrame(spark_df)
print("Total elements = %d"%final_data_df.shape[0])
print("Type of frame = %s"%type(final_data_df))

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting train data into train, test, and validation datasets

# COMMAND ----------

import random
import numpy as np
def test_train_split_timeseries_data(data_df, split_ratio, random_ratio= False):
    
    train_split = split_ratio
    
    # Find the unique store ids
    unique_elements = data_df['store'].unique().to_numpy()
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
temp_x, temp_y, test_x, test_y, df_train, df_test = test_train_split_timeseries_data(final_data_df,0.8)
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

data_path = os.path.join(os.getcwd(),'train.csv')

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



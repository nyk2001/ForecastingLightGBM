# Databricks notebook source
# MAGIC %md
# MAGIC # Install LightGBM Library

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %pip install holidays

# COMMAND ----------

# MAGIC %pip install meteostat

# COMMAND ----------

# MAGIC %pip install 'protobuf<=3.20.1' --force-reinstall

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
# MAGIC # Load data in PySpark

# COMMAND ----------

import os
import pyspark.pandas as ps

# Assuming we already have a table sales_train_data in default Hive database
spark_df=spark.sql("select * from sales_train_data")
spark_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Train Data Split

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import DoubleType, StructField

##############################################################################
# Verifies train and test data distributions - Only for debug for this use case
##############################################################################
# Inputs:
#    --> Pyspark Data frame 
# Output:
#    --> True if train data distribution is around 80% and test data distribution is around 20%
def verify_data_distributions(input_dataframe):
    
    #data_df = input_dataframe.to_pandas_on_spark() - too slow for debugging
    data_df = input_dataframe.toPandas()
    
    # Find unique store and item elements
    unique_store_elements = data_df['store'].unique()
    unique_item_elements = data_df['item'].unique()

    output =list()
    index=0
    for each_item_element in unique_item_elements:    
        output.append([each_item_element])
        for each_store_element in unique_store_elements:    
            # Extract train and test distributions for each store and corresponding item
            temp_df = data_df[(data_df['store']==int(each_store_element)) & (data_df['item']==int(each_item_element))].copy()
            output[index].extend([temp_df[temp_df['Dataset']=='Train'].shape[0]])
            output[index].extend([temp_df[temp_df['Dataset']=='Test'].shape[0]])
            output[index].extend([100.0*(temp_df[temp_df['Dataset']=='Train'].shape[0]/temp_df.shape[0])])
            output[index].extend([100.0*(temp_df[temp_df['Dataset']=='Test'].shape[0]/temp_df.shape[0])])
        index+=1
    
    # Generate stats dataframe for validation
    stats_df= ps.DataFrame(output,columns=["Item",
                                           'Store1_TR', 'Store1_TST','Store1_PTR','Store1_PTST', 'Store2_TR',  'Store2_TST', 'Store2_PTR', 'Store2_PTST',
                                           'Store3_TR', 'Store3_TST','Store3_PTR','Store3_PTST', 'Store4_TR',  'Store4_TST', 'Store4_PTR', 'Store4_PTST',                                       
                                           'Store5_TR', 'Store5_TST','Store5_PTR','Store5_PTST', 'Store6_TR',  'Store6_TST', 'Store6_PTR', 'Store6_PTST',
                                           'Store7_TR', 'Store7_TST','Store7_PTR','Store7_PTST', 'Store8_TR',  'Store8_TST', 'Store8_PTR', 'Store8_PTST',
                                           'Store9_TR', 'Store9_TST','Store9_PTR','Store9_PTST', 'Store10_TR', 'Store10_TST','Store10_PTR','Store10_PTST'])
    
    # Going through each train and test distributions
    for each_store_element in unique_store_elements:    
        temp_train_df= stats_df[(stats_df['Store%d_PTR'%each_store_element]<=(80-0.3)) | (stats_df['Store%d_PTR'%each_store_element]>=(80+0.3))]
        temp_test_df= stats_df[(stats_df['Store%d_PTST'%each_store_element]<=(20-0.3)) | (stats_df['Store%d_PTST'%each_store_element]>=(20+0.3))]
        
        # Return false if distributions are not correct
        if temp_train_df.shape[0]>0 or temp_test_df.shape[0]>0:
            return False
    return True
    
        
##############################################################################
# Splits timeseries data into train and test
##############################################################################
# Inputs:
#    --> Pyspark Data frame 
#    --> Alpha values, such as 1 , 0.95
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Ewm column suffix, such as lag
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pyspark dataframe with new columns

def test_train_split_timeseries_data(df, group_cols, split_ratio):
    df.orderBy("store","item","date")
    df= df.withColumn("Dataset", F.lit("Train").cast('string'))
    
    # Using Pyhton function to compute stats for each group
    def split_dataset(pdf):
        pdf = pdf.sort_values(['date'])        
        train_rows = int (pdf.shape[0]*split_ratio)
        
        # Get the reference date
        reference_date = pdf.iloc[train_rows]['date']
        
        # Assign test to other rows
        pdf.loc[pdf['date']>reference_date, 'Dataset']='Test'
        
        return pdf

    # Split and apply 
    df= df.groupby(group_cols).applyInPandas(split_dataset, schema= df.schema)
    
    # Validating data distribution - onyl for debugginf the code
    print(f"No Error found in distribution = {verify_data_distributions(df)}")
    
    from pyspark.sql.functions import col
    df_train= df.filter(col("Dataset")=='Train')
    df_test= df.filter(col("Dataset")=='Test')
    
    df_train= df_train.drop(df.Dataset)
    df_test =df_test.drop(df.Dataset)
    
    df_train.orderBy("store", "item", "date")
    df_test.orderBy("store", "item", "date")
                       
    return df_train, df_test

# Split dataset into 80% train and 20% testing
df_train, df_test=test_train_split_timeseries_data(spark_df, ["store","item"], 0.8)
print(f"Total rows of training dataset = {df_train.count()}")
print(f"Total rows of test dataset = {df_test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert to pandas - for debugging only

# COMMAND ----------

# Convert to Pandas - Pandas API is very slow
df_train_pandas = df_train.toPandas()
df_test_pandas = df_test.toPandas()

# Sorting values
df_train_pandas = df_train_pandas.sort_values(["store","item","date"])
df_test_pandas = df_test_pandas.sort_values(["store","item","date"])

df_train_pandas.reset_index(drop=True, inplace=True)
df_test_pandas.reset_index(drop=True, inplace=True)

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

from datetime import date
import holidays
    
    
##############################################################################
# Generating weather information of Australia, such as min, max and average temp
##############################################################################
# Input
#   --> Year
#   --> Month
#   --> Day
# Output
#   --> Min, Max and Average temperatures
def get_weather_stats(year, month, day):
    
    from meteostat import Stations, Daily
    from meteostat import Stations
    from datetime import datetime
    
    # Get stations
    stations = Stations()
    stations = stations.region('AU', 'VIC')
    
    # Fetch the nearest station
    station = stations.fetch(1)
    data = Daily(station, start = datetime(year, month, day), end = datetime(year, month, day))

    # Fetch temperatures
    data = data.fetch()
    
    return data.tmin[0], data.tmax[0], data.tavg[0]

    
##############################################################################
# Generating information about Australia's public holiday - victoria state
##############################################################################
# Input
#   --> date value
# Output
#   --> 1 if it is a public holiday, 0 otherwise
def get_public_holiday(x):
    state ='VIC'
    # Get public holidays for Australia State
    public_holidays = holidays.Australia(state= state)
    holiday = public_holidays.get(x)
    if holiday is None:
        return 0
    else:
        return 1 
    
##############################################################################
# Creates time features using pandas data frame api
##############################################################################    
# Input
#   --> Pandas API Dataframe
# Output
#   --> Dataframe with various time outputs
def create_time_features_pandas(df):
    
    df['date'] = ps.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year    
    df['week_of_year'] = df['date'].dt.week.astype(int) # Which week of the corresponding year
    df['day_of_week'] = df['date'].dt.dayofweek # Which day of the corresponding week of the each month
    df["is_weekend"] = (df.date.dt.weekday // 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int) # Is it starting of the corresponding month
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int) # Is it ending of the corresponding month    
    df['public_holiday']= df['date'].apply(get_public_holiday)
    #df[['min_temperature','max_temperature','avg_temperature']]= df.apply(lambda row: get_weather_stats(row["year"], row["month"], row["day"]), axis=1) 
    
    return df

# Generate time features based on the date column
df_train_pandas = create_time_features_pandas(df_train_pandas)
df_test_pandas = create_time_features_pandas(df_test_pandas)
df_train_pandas.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Using PySpark

# COMMAND ----------

from pyspark.sql.types import DateType, FloatType, IntegerType
from pyspark.sql.functions import year, month, dayofmonth, to_date, col, weekofyear, dayofweek,last_day,to_timestamp
from pyspark.sql.functions import udf
from meteostat import Stations, Daily
from meteostat import Stations
from datetime import datetime
import holidays
##############################################################################
# Generating weather information of Australia, such as min, max and average temp
##############################################################################
# Input
#   --> Year
#   --> Month
#   --> Day
# Output
#   --> Min, Max and Average temperatures
@udf
def get_weather_stats(x,y,z):
    
    # Get stations
    stations = Stations()
    stations = stations.region('AU', 'VIC')
    
    # Fetch the nearest station
    station = stations.fetch(1)
   
    # Fetch temperatures
    data = Daily(station, start = datetime(x, y, z), end = datetime(x, y, z))
    data = data.fetch()
    
    # Temperature stats
    #data.tmin[0], data.tmax[0], data.tavg[0]
    return data.tavg[0]

##############################################################################
# Generating information about public holiday - victoria state
##############################################################################
# Input
#   --> date value
# Output
#   --> 1 if it is a public holiday, 0 otherwise
@udf
def get_public_holiday(x):
    public_holidays = holidays.Australia(state= 'VIC')
    holiday = public_holidays.get(x)
    if holiday is None:
        return 0
    else:
        return 1 

##############################################################################
# Generating time features
##############################################################################
# Input
#   --> Pandas API Dataframe
# Output
#   Dataframe with various time outputs
def create_date_features_pyspark(df):
    public_holidays = holidays.Australia(state= 'VIC')
    
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
    df = df.withColumn("public_holiday", get_public_holiday(df['date']))
    #df = df.withColumn("avg_temperatue",get_weather_stats(df['year'], df['month'], df['day']))

    return df


 
# Create pyspark data frame 
df_train= create_date_features_pyspark(df_train)
df_test= create_date_features_pyspark(df_test)

df_train.orderBy("store","item","date")
df_test.orderBy("store","item","date")

df_train.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Misc Functions

# COMMAND ----------

##############################################################################
# Save table to a default schema 
##############################################################################
# Input
#  --> Dataframe. Can be pandas or spark
#  --> Table name
# Output
#  --> None
def write_data_hive(df, table_name):
    from pyspark.sql import DataFrame

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
##############################################################################
# Creating lagged features
##############################################################################
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
df_train_pandas = create_lagged_features_pandas(df_train_pandas,lag_window,["store", "item"],"sales", "lag")
df_test_pandas = create_lagged_features_pandas(df_test_pandas,lag_window,["store", "item"],"sales", "lag")
print("Total elements = %d"%df_test_pandas.shape[0])
df_test_pandas.head(10)   

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
import numpy as np

##############################################################################
# Creating lagged features
##############################################################################
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
df_train = create_lagged_features_pyspark(df_train, lag_window,['store','item'],"sales","lag")
df_train= df_train.orderBy("store","item","date")
df_train.show()

df_test = create_lagged_features_pyspark(df_test, lag_window,['store','item'],"sales","lag")
df_test= df_test.orderBy("store","item","date")
df_test.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Comparing columns between Pandas and PySpark dataframe

# COMMAND ----------

# Comapring columns
col_list=['sales_lag_1','sales_lag_3','sales_lag_5','sales_lag_7']
comapre_data(df_train,df_train_pandas,['store', 'item','date'], col_list)

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

##############################################################################
# Creating rolling features
##############################################################################
# Inputs:
#    --> Pandas Data frame 
#    --> Windows values, such as 3 , 5 , 7
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Roll column suffix, such as lag
#    --> List of operations such as sum , mean, or stfcan be sum or mean
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pandas dataframe with new columns
def create_window_rolling_features(df, window_sizes,group_cols, target_col_name,rolling_suffix, operations, random_noise= False):
    
    for each_ooperation in operations: # Loop through each operation
        for window_size in window_sizes:# Loop through each window size
            if each_ooperation == "sum":
                df["sales" + "_" +rolling_suffix+"_" + each_ooperation + "_" + str(window_size)] = df.groupby(group_cols)[target_col_name].\
                    transform(lambda x: x.rolling(window=window_size, center= True,min_periods=1).sum())
            if each_ooperation =="mean":
                df["sales" + "_" + rolling_suffix+"_"+ each_ooperation + "_" + str(window_size)] = df.groupby(group_cols)[target_col_name].\
                    transform(lambda x: x.rolling(window=window_size,center= True,min_periods=1).mean())
            if each_ooperation =="std":
                df["sales" + "_" + rolling_suffix+"_"+ each_ooperation + "_" + str(window_size)] = df.groupby(group_cols)[target_col_name].\
                    transform(lambda x: x.rolling(window=window_size,center= True,min_periods=1).std())
        
    return df

# Using two window sizes
windows_sizes = [3,5,7] 
#pandas_df_api_v2 = pandas_df_api_v2.sort_values(by= ["store","item","date"], ascending=True)
df_train_pandas = create_window_rolling_features(df_train_pandas, windows_sizes,["store", "item"], "sales", "roll", ["sum", "mean", "std"])
df_test_pandas = create_window_rolling_features(df_test_pandas, windows_sizes,["store", "item"], "sales", "roll", ["sum", "mean", "std"])

# COMMAND ----------

import pyspark.pandas as ps
from pyspark.sql import *
ps.set_option('compute.ops_on_diff_frames', True)

##############################################################################
# Creating rolling features
##############################################################################
# Inputs:
#    --> Pyspark Data frame 
#    --> Windows values, such as 3 , 5 , 7
#    --> Cols used for grouping, can be a single column or multiple columns
#    --> Target columm name, such as sales
#    --> Roll column suffix, such as lag
#    --> List of operations such as sum , mean, or stfcan be sum or mean
#    --> Random noise True or False - adds random noise to variables (default is False)
# Output:
#    --> Pyspark dataframe with new columns

def create_rolling_features(df, window_sizes,group_cols, target_col_name,rolling_suffix, operations, random_noise= False):
    
    # Calculating days
    days = lambda i: i * 86400
    for each_ooperation in operations: # Loop through each operation
        for window_size in window_sizes:        # Loop through each window size
            
            # Computing window delta
            window_delta = int(window_size/2)

            # Creating window for roll feature
            roll_window = Window().partitionBy(group_cols).orderBy(F.col("timestamp").cast('long')).rangeBetween(-days(window_delta), days(window_delta))    

            # Generate new column
            if each_ooperation == "sum":
                df = df.withColumn(f"{target_col_name}_{rolling_suffix}_{each_ooperation}_{window_size}", F.sum(target_col_name).over(roll_window))
            if each_ooperation == "mean":
                df = df.withColumn(f"{target_col_name}_{rolling_suffix}_{each_ooperation}_{window_size}", F.avg(target_col_name).over(roll_window))
            if each_ooperation == "std":
                df = df.withColumn(f"{target_col_name}_{rolling_suffix}_{each_ooperation}_{window_size}", F.stddev(target_col_name).over(roll_window))

    return df

    
# Using two window sizes
windows_sizes = [3,5,7] 

# Generate mean and sum rolling features
df_train = create_rolling_features(df_train, windows_sizes,["store","item"], "sales", "roll", ["sum", "mean", "std"])
df_train = df_train.orderBy("store","item","date")

df_test = create_rolling_features(df_test, windows_sizes,["store","item"], "sales", "roll", ["sum", "mean", "std"])
df_test = df_test.orderBy("store","item","date")

df_test.limit(20).toPandas().head(10)  


# COMMAND ----------

# MAGIC %md
# MAGIC Comapring columns between Pandas Dataframe and PySpark

# COMMAND ----------

# Comapring columns
col_list=['sales_roll_mean_3','sales_roll_mean_5','sales_roll_mean_7','sales_roll_sum_3','sales_roll_sum_5','sales_roll_sum_7','sales_roll_std_3','sales_roll_std_5','sales_roll_std_7']
comapre_data(df_train,df_train_pandas,['store', 'item','date'], col_list)

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

##############################################################################
# Creating Exponentially Weighted Mean Features
##############################################################################
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
        col_name = target_col_name + "_" + ewm_suffix  + "_" + str(alpha).replace('.','')
        df[col_name] = \
                df.groupby(group_cols)[target_col_name].transform(lambda x: x.ewm(alpha=alpha).mean())
        
    return df

# In here we have two combinations : alphas and lags
alphas = [0.6, 0.85]

df_train_pandas = create_ewm_features(df_train_pandas, alphas, ["store","item"], "sales", "ewm")
df_train_pandas = df_train_pandas.sort_values(["store","item","date"])

df_test_pandas = create_ewm_features(df_test_pandas, alphas, ["store","item"], "sales", "ewm")
df_test_pandas = df_test_pandas.sort_values(["store","item","date"])

df_test_pandas.head(20) 

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import DoubleType, StructField

##############################################################################
# Creating Exponentially Weighted Mean Features
##############################################################################
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
        col_name = f"{target_col_name}_{ewm_suffix}_{str(alpha).replace('.','')}"
        df = df.withColumn(col_name, F.lit(None).cast('double'))
        
        # Using Pyhton function to compute ema stats for each group
        def ema(pdf):
            pdf[col_name] = pdf[target_col_name].ewm(alpha=alpha).mean()
            
            return pdf
        
        # Split and apply 
        df= df.groupby(group_cols).applyInPandas(ema, schema = df.schema)
    
    return df

alphas= [0.6, 0.85]    
df_train=create_ewm_features(df_train, alphas, ["store","item"], "sales", "ewm")
df_train = df_train.orderBy("store","item","date")
df_test=create_ewm_features(df_test, alphas, ["store","item"], "sales", "ewm")
df_test = df_test.orderBy("store","item","date")

# COMMAND ----------

# Comapring columns
col_list=['sales_ewm_06','sales_ewm_085']
comapre_data(df_train,df_train_pandas,['store', 'item','date'], col_list)

# COMMAND ----------

#write_data_hive(final_data_df, "sale_train_final_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split train into Train and Validation datasets

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting train data into train, and validation datasets

# COMMAND ----------

df_test_set= df_test
# Split dataset into 80% train and 20% testing
df_train_set, df_validation_set =test_train_split_timeseries_data(df_train, ["store","item"], 0.8)
print(f"Total rows of training dataset = {df_train_set.count()}")
print(f"Total rows of validation dataset = {df_validation_set.count()}")
print(f"Total rows of test dataset = {df_test.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare datasets for LightGBM training and testing

# COMMAND ----------

##############################################################################
# Extract features and target variable as two frames
##############################################################################
# Inputs:
#    --> Pyspark Data frame 
#    --> Target or output variable 
#    --> List of cols to be excluded
# Output:
#    --> Pyspark dataframe with features
#    --> Pyspark dataframe with output targer variable
def prepare_dataset_sets(df, target, exclude_cols= []):
     
    # Excluding the target variable from trianing set along with any other unwanted columns
    total_columns_exclude = exclude_cols+  [target]
    
    # Finding training column list
    x = [i for i in df.columns if i not in total_columns_exclude]    
    y = target
       
    return df[x], df[[y]]
  
# Get features and target dataframes for training set    
train_x, train_y = prepare_dataset_sets(df_train_set, 'sales',['date','timestamp'])

# Get features and target dataframes for validation set    
validation_x, validation_y = prepare_dataset_sets(df_validation_set, 'sales',['date','timestamp'])

# Get features and target dataframes for test set    
test_x, test_y = prepare_dataset_sets(df_test, 'sales',['date','timestamp'])

# COMMAND ----------

# Count of rows in datasets
train_x.count(), validation_x.count(), test_x.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model - Simple no tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Model preparation

# COMMAND ----------

##############################################################################
# Get LightGBM Model Parameters
##############################################################################
# Inputs:
#    --> Custom search parameters
# Output:
#    --> Search parameter space to be used by the algorithm
def get_model_parameters(search_params):
    fixed_params={
            'nthread': 10,
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression_l1',
            'metric': ['mape','rmse'], 
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

##############################################################################
# Get LightGBM Model Parameters
##############################################################################
# Inputs:
#    --> Train_x : PySpark dataframe
#    --> Train_y : PySpark dataframe
#    --> Test_x  : PySpark dataframe
#    --> Test_y  : PySpark dataframe
#    --> Fixed number of ierations
#    --> Parameters Space
# Output:
#    --> Trained Model
#    --> Validation Mape Score
def train_model(train_x,train_y,test_x,test_y,iterations, parameters_space):
    
    # Get parameters from models
    parameters = get_model_parameters(parameters_space)
    
    # Create datasets (converting pyspark to numpy)
    lgb_train = lgb.Dataset(np.array(train_x.collect()), np.array(train_y.collect()))
    lgb_valid = lgb.Dataset(np.array(test_x.collect()), np.array(test_y.collect()))
    
    # Model crated
    model = lgb.train(parameters, 
                      lgb_train, 
                      iterations, 
                      valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, 
                      verbose_eval=50
                     )
    
    # Return the best map score and the model
    score_mape = model.best_score['valid_1']['mape']
    score_rmse = model.best_score['valid_1']['rmse']
    
    return model, score_mape, score_rmse


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Use MLFlow for model tracking

# COMMAND ----------

# MAGIC %md
# MAGIC Model training

# COMMAND ----------

#The Model signature defines the schema of a model's inputs and outputs. Model inputs and outputs can be either column-based or tensor-based.
from mlflow.models.signature import infer_signature
from mlflow.tracking.client import MlflowClient
import mlflow
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
    mlflow.lightgbm.autolog(log_input_examples=True, log_model_signatures=True, log_models=True) # enabling autolog
    model , score_mape, score_rmse = train_model(train_x,train_y,validation_x,validation_y,5000, {})  # training model
    
    print('The best MAPE for validation = %0.3f'%score_mape)
    print('The best RMSE for validation = %0.3f'%score_rmse)
    
    # Saving feature importances
    ax = lgb.plot_importance(model, max_num_features=10)    
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    # Extracting model signature
    signature = infer_signature(np.array(train_x.collect()), model.predict(np.array(train_x.collect())))
    
    mlflow.log_metric("mape", score_mape)
    mlflow.log_metric("rmse", score_rmse)
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

# MAGIC %md
# MAGIC Test performance on test data

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

model_path = f"runs:/{run_id}/model"
    
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_path)

# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(np.array(test_x.collect()))
rmse = mean_squared_error(np.array(test_y.collect()), predictions)
mape = mean_absolute_percentage_error(np.array(test_y.collect()), predictions)
print("RMSE on test dataset is = %0.3f"%rmse)
print("MAPE on test dataset is = %0.3f"%mape)



# COMMAND ----------

# MAGIC %md
# MAGIC Performance for individual stores

# COMMAND ----------

# Computing RMSE and MAPE for each individual store
test_df = ps.concat([ps.DataFrame(test_y), ps.DataFrame(test_x)], axis=1)
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
            'metric': ['mape','rmse'], 
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
    
    iterations =2000
    
    # Model crated
    model = lgb.train(parameters, lgb_train, 
                      iterations, valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, verbose_eval=50)
    
    # Return the best map score and the model
    mape = model.best_score['valid_1']['mape']
    rmse = model.best_score['valid_1']['rmse']
    
    
    return mape + rmse

# COMMAND ----------

from hyperopt import hp, fmin,tpe, Trials
import mlflow

# Create datasets
lgb_train = lgb.Dataset(np.array(train_x.collect()),np.array(train_y.collect()))
lgb_valid = lgb.Dataset(np.array(validation_x.collect()),np.array(validation_y.collect()))

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

num_evals = 10
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
    forecast_model , score_mape, score_rmse = train_model(train_x,train_y,validation_x,validation_y,3000, seearch_space)
    
    # Log model
    mlflow.lightgbm.log_model(forecast_model, "lightgbm-model", input_example=train_x.limit(5).toPandas())
    
    mlflow.log_metric("val mape", score_mape)
    print('The best MAPE for validation = %0.3f'%score_mape)
    mlflow.log_metric("val rmse", score_rmse)
    print('The best RMSE for validation = %0.3f'%score_rmse)
    
    ax = lgb.plot_importance(forecast_model, max_num_features=10)
    fig = plt.gcf()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_hyperparameter['max_depth'])
    mlflow.log_param("numLeaves", best_hyperparameter['num_leaves'])
    mlflow.log_param("learningRate", best_hyperparameter['lr'])
    mlflow.log_param("bagging_fraction", best_hyperparameter['bagging_fraction'])
    mlflow.log_param("feature_fraction", best_hyperparameter['feature_fraction'])
    
    # Get run ids
    run = mlflow.active_run()
    run_id = run.info.run_id
    

# COMMAND ----------

# Computing RMSE and MAPE for each individual store
test_df = ps.concat([ps.DataFrame(test_y), ps.DataFrame(test_x)], axis=1)
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

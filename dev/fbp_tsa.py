## FaceBook Prophet - Time Series Analysis

# Author: Samuel Au
# Date: 2025-03-15

from collections import defaultdict
import csv
from math import ceil, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
  mean_absolute_percentage_error,
  mean_absolute_error,
  mean_squared_error,
  r2_score
)

import logging
logging.getLogger("prophet.plot").disabled = True  # disables INFO log for prophet failing to import plotly
logger = logging.getLogger('cmdstanpy')            # disables INFO log for cmdstanpy
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

from prophet import Prophet  # need prophet import after logging to disable INFO log output

# ----------------------------------------------------------------------------------------------- #
# START USER DEFINED FUNCTIONS
# ----------------------------------------------------------------------------------------------- #

def read_csv_iata(filepath):
  '''
  Uses Processed CSV, filepath = "data/processed-data.csv"
  Output: {('airport_iata_1', 'airport_iata_2'): {'date': 'yyyy-qq', 'fare': 'XX.XX'}}
  '''
  output = defaultdict(lambda: defaultdict(list))
  with open(filepath, 'r') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
      # print(row['year'], row['quarter'], row['airport_iata_1'], row['airport_iata_2'], row['fare'])
      output[(row['airport_iata_1'], row['airport_iata_2'])]['date'].append(f"{row['year']}-Q{row['quarter']}")
      output[(row['airport_iata_1'], row['airport_iata_2'])]['fare'].append(row['fare'])

  return output

def read_csv(filepath):
  '''
  Uses Raw CSV, filepath = "data/raw-data.csv"
  Output: {('airport_1', 'airport_2'): {'date': 'yyyy-qq', 'fare': 'XX.XX'}}
  '''
  output = defaultdict(lambda: defaultdict(list))
  with open(filepath, 'r') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
      # print(row['Year'], row['quarter'], row['airport_iata_1'], row['airport_iata_2'], row['fare'])
      output[(row['airport_1'], row['airport_2'])]['date'].append(f"{row['Year']}-Q{row['quarter']}")
      output[(row['airport_1'], row['airport_2'])]['fare'].append(row['fare'])

  return output

def process_df(data: pd.DataFrame, cols2keep: list = None) -> pd.DataFrame:
  '''
  Processes raw csv DF by concatenating str cols for easier filtering and date transformations.
  Combines cols: ['year'] + '-Q' + ['quarter'] = '2024' + '-Q' + '1' => ['date'] = '2024-Q1'
  Combines cols: ['airport_1'] + '-' + ['airport_1'] = 'BOS' + '-' + 'LGA' => ['route'] = 'BOS-LGA'
  :return: new cols=['date', 'route'] of type=[str, str]
  '''
  df = data
  df['date'] = df['year'].str.cat(df['quarter'], sep='-Q')  # example output '2024-Q1'
  df['route'] = df['airport_1'].str.cat(df['airport_2'], sep='-')  # example output 'BOS-LGA'
  return df[cols2keep] if cols2keep else df

def filter_routes(data: pd.DataFrame,
                  air1: str, air2: str,
                  min_rows: int = 0,
                  sorted: bool = False,
                  area_code: bool = False) -> pd.DataFrame | None:
  '''
  Filters `data` DF ['route'] for an airport route that starts with `air1` and ends at `air2`.
  Filtered `data` must have at least `min_rows`, otherwise return None.
  Assumes `data` is not sorted and returns a sorted DF based on col='date' of type str.
  If `data` is known to be sorted, then set `sorted` to be 'True'.
  If `area_code` then aggregate (average) all airports inside area by grouping on the date col

  :params:
    data = pd.Dataframe({'date': ['2024-Q1', '2023-Q4', ...], 'route': ['BOS-LGA', 'SFO-SAN', ...]})
    air1 | air2 = 'ATL'  # 3 letter airport code
    min_rows = minimum number of rows  # 50 is desired size of DF for time series analysis
    sorted = boolean indicating if input dataframe is sorted by date or not
    area_code = boolean indicating that 3 letter code is metropolitan area code, not airport
  :return: 
    pd.Dataframe({'date': ['2023-Q3', '2024-Q1', ...], 'y': ['BOS-LGA', 'BOS-LGA', ...]})
    None if pd.DataFrame.shape[0] < min_rows
  '''
  mask = (data['route'] == f'{air1}-{air2}')
  df = data[mask]
  if area_code:
    df['fare'] = df['fare'].astype('float64')
    df = df.groupby('date')['fare'].mean()
    df = df.reset_index(drop=True)
    df.columns = ['date', 'fare']
  if df.shape[0] < min_rows:
    return None
  return df.sort_values(by='date').reset_index(drop=True) if not sorted else df

def find_longest_timeseq(data: pd.DataFrame, ycol: str, min_rows: int = 0) -> pd.DataFrame | None:
  '''
  Finds the longest sequence of quarterly dates that is at least of size `limit`.
  Returns a pd.DF that is in format for FB Prophet e.g. cols = ['ds', 'y']
    where 'ds' is datestamp [datetime] and 'y' is column to forecast [float]

  :params:
    data = pd.Dataframe({'date': ['2023-Q4', '2024-Q1', ...], 'fare': [1.2, 3.4, ...]})
    ycol = 'fare'  # column to forecast
    min_rows = 50  # desired size of DF for time series analysis
  :return: 
    pd.Dataframe({'ds': ['2023-12-31', '2024-03-31', ...], 'y': [1.2, 3.4, ...]})
  '''
  df = data.copy()

  # Perform Datetime Transformation for Mask
  df['period'] = pd.PeriodIndex(df['date'], freq='Q')  # freq='Q' expects '%Y-%q' format, returns type period[Q-DEC]
  df['shift'] = df['period'].shift(periods=1, fill_value=(df['period'].min() - 1))  # shift down 1 quarter

  # Create Mask to Filter for Longest Subsequence
  df['bool'] = (df['shift'] + 1 != df['period'])  # check if next date is exactly 1 qtr ahead
  df['mask'] = df['bool'].cumsum()  # 'True' indicate break in sequence; cumsum groups breaks

  # Create column in datetime64[ns] format for FB Prophet
  df['ds'] = df['period'].dt.to_timestamp(freq='Q')  # outputs end of qtr e.g. 1996Q1 -> 1996-03-01
  # Note use `dt` accessor to avoid TypeError: unsupported Type RangeIndex during conversion

  # Result DF
  res = df.loc[df['mask'] == df['mask'].value_counts().idxmax(), ['ds', ycol]]
  res = res.reset_index(drop=True)
  res = res.rename(columns={ycol: 'y'})
  res['y'] = res['y'].astype('float64')

  # Return Result DF if count_rows >= limit
  return res if res.shape[0] >= min_rows else None

def endsIn2024(data: pd.DataFrame) -> bool:
  '''
  Return True if time-series DF ends in 2024.
  Assumes DF is in form for FB Prophet with cols ['ds', 'y'] of type [datetime[ns], float64]

  :param: data = pd.Dataframe({'ds': ['2023-12-31', '2024-03-31', ...], 'y': [1.2, 3.4, ...]})
  '''
  return data['ds'].max().year == 2024

def train_test_split(data: pd.DataFrame, test_split: int | float | str) -> pd.DataFrame:
  '''
  Split `data` DF based on `test_split` and returns two DFs "train_set" and "test_set"

  :params:
    data = pd.Dataframe({'ds': [2022-06-30, ...], ...}) where date is type datetime[ns]
    test_split = int -> get last n rows as "test_set"; int must be btw 0 and len(data)
    test_split = float -> get last percent rows as "test_set"; float must be btw 0 and 1
    test_split = str -> Assume str is date 'YYYY-MM-DD' and split on date
  :return:
    train_set, test_set 
  '''
  if isinstance(test_split, int):
    assert 0 < test_split < data.shape[0]
    return data.iloc[:-test_split], data.iloc[-test_split:]
  elif isinstance(test_split, float):
    assert 0 < test_split < 1
    index = ceil(data.shape[0] * test_split)
    return data.iloc[:-index], data.iloc[-index:]
  elif isinstance(test_split, str):
    mask = (data['ds'] <= test_split)
    return data[mask], data[~mask]

def plot_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> plt:
  '''
  Creates Scatterplot of Training and Test Data

  :params:
    train_df | test_df = pd.Dataframe({'ds': ['2024-03-31', ...], 'y': [1.2, ...]})
  :return:
    matplotlin.pyplot
  '''
  plt.figure(figsize=(10, 6))
  plt.scatter(train_df['ds'], train_df['y'], label='Train Data', color='blue')
  plt.scatter(test_df['ds'], test_df['y'], label='Test Data', color='red')
  plt.xlabel('Time')
  plt.ylabel('y')
  plt.title('Scatter Plot of Train and Test Data')
  plt.legend()
  plt.show()

def fbp_predict_future(model: Prophet,
                        data: pd.DataFrame,
                        n: int,
                        plot_fcst: bool = False,
                        plot_comp: bool = False,
                        unc: bool = True) -> pd.DataFrame:
  '''
  Use FB Prophet to create prediction into the future by `n` periods with optional plotting

  :params:
    model = Prophet(...) with any additional parameters added to it
    data = pd.Dataframe({'ds': ['2024-03-31', ...], 'y': [1.2, ...]}, dtype = [datetime[ns], float64]
    n = 8  # plot 8 periods or quarters into future
    plot_fcst = toggle plotting of future forecasts with historical data
    plot_comp = toggle components of model model e.g. trend, seasonality, etc.
    unc = toggle plotting of uncertainty
  :return:
    forecast = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', ...]
    Optional: matplotlin.pyplot
  '''
  # Prophet Model
  model = model.fit(data)  # train model on DF

  # Make Future DF to Store Predictions
  future = model.make_future_dataframe(periods=n, freq='QE', include_history=False)

  # Make Predictions on Future DF
  forecast = model.predict(future)
  
  # Optional: Show Forecast Plot
  if plot_fcst:
    fig1 = model.plot(forecast, uncertainty=unc)
    plt.show()
  
  # Optional: Show Component Plot
  if plot_comp:
    fig2 = model.plot_components(forecast, uncertainty=unc)
    plt.show()

  return forecast

def fbp_predict_test(model: Prophet, train: pd.DataFrame, test: pd.DataFrame, plot_act_vs_pred: bool = False) -> pd.DataFrame:
  '''
  Use FB Prophet to create prediction on `test` data using `train` data with optional plotting

  :params:
    model = Prophet(...) with any additional parameters added to it
    tran | test = pd.Dataframe({'ds': ['2024-03-31', ...], 'y': [1.2, ...]}, dtype = [datetime[ns], float64]
    plot = toggle plotting of actual `test` data vs predicted test data
  :return:
    forecast = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', ...]
    Optional: matplotlin.pyplot
  '''
  # Prophet Model
  model = model.fit(train)  # train model on DF

  # Make Predictions on Test DF
  test_forecast = model.predict(test)

  # Optional Plotting for Actual vs Predicted Test Data
  if plot_act_vs_pred:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(test_df['ds'], test_df['y'], label='Actual', color='r')
    fig = model.plot(test_forecast, ax=ax)
    # ax.set_xlim([pd.to_datetime('2022-06-01', format = '%Y-%m-%d'),
    #             pd.to_datetime('2024-04-30', format = '%Y-%m-%d')])
    ax.set_xbound(lower=test_df['ds'].iloc[1] - pd.tseries.offsets.DateOffset(months=4),
                  upper=test_df['ds'].iloc[-1] + pd.tseries.offsets.DateOffset(months=1))
    plt.legend()
    plt.show()

  return test_forecast

# ----------------------------------------------------------------------------------------------- #
# END USER DEFINED FUNCTIONS
# ----------------------------------------------------------------------------------------------- #


if __name__ == "__main__":

  # --------------------------------------------------------------------------------------------- #
  # Import Data
  # --------------------------------------------------------------------------------------------- #
  filepath = "data/raw-data.csv"
  cols2read = ['year', 'quarter', 'airport_1', 'airport_2', 'fare']
  # data = read_csv(filepath)  # using csv module and raw_data
  # data = read_csv_iata(filepath)  # using csv module and processed_data
  # data = pd.read_csv(filepath, usecols=cols2read, dtype=object, engine='c')  # read without inferring dtypes
  data = pd.read_csv(filepath,
                      names=['tbl', 'year', 'quarter', 'citymarketid_1', 'citymarketid_2', 'city1',
                              'city2', 'airportid_1', 'airportid_2', 'airport_1', 'airport_2', 'nsmiles',
                              'passengers', 'fare', 'carrier_lg', 'large_ms', 'fare_lg', 'carrier_low',
                              'lf_ms', 'fare_low', 'geocoded_city1', 'geocoded_city2', 'tbl1apk'
                            ],
                      usecols=cols2read,
                      dtype={'year': 'str', 'quarter': 'str', 'airport_1': 'str', 'airport_2': 'str',
                              'fare': 'str'
                            },
                      header=0,
                      engine='c')  # read with manual dtypes, notes 'str' = 'object'
  # print(data.head())
  # print(data.dtypes)

  # --------------------------------------------------------------------------------------------- #
  # Clean Data
  # --------------------------------------------------------------------------------------------- #
  cols2keep = ['date', 'route', 'fare']
  data = process_df(data, cols2keep) # clean and process data into better format
  # print(data.head())

  # --------------------------------------------------------------------------------------------- #
  # TEST CODE FUNCTIONS FOR 1 UNIQUE ROUTE (<1s run)
  # --------------------------------------------------------------------------------------------- #

  ## Find Route BOS -> LGA and its Longest Subsequence of Dates with Min Count 50
  ## Note: 10 Busiest airport by passengers are ORD, DFW, BOS, LAX, MDW, ATL, IAH, DEN, LAS, DTW
  # df = filter_routes(data, 'MDW', 'CVG')  # returns sorted_df of a direct flight
  # df = find_longest_timeseq(df, ycol='fare', min_rows=50)
  # assert endsIn2024(df) == True  # check if DF has ['ds'] that ends in 2024

  # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  #   print(df)
  # print(df.dtypes)

  ## Train / Test Split
  # df, ignore = train_test_split(df, test_split='2020-03-21')  # split df on COVID
  # train_df, test_df = train_test_split(df, test_split=8)  # test set is last 8 rows
  # print(train_df, test_df)
  # plot_train_test(train_df, test_df)  # scatterplot

  ## FB Prophet
  # m = Prophet(seasonality_mode='additive', yearly_seasonality='auto')  # default params
  # m = Prophet(seasonality_mode='multiplicative', yearly_seasonality='auto')

  # future_forecast = fbp_predict_future(model=m, data=df, n=8, plot_fcst=False, plot_comp=True)

  # test_forecast = fbp_predict_test(model=m, train=train_df, test=test_df, plot_act_vs_pred=True)

  # mape = mean_absolute_percentage_error(y_true=test_df['y'], y_pred=test_forecast['yhat'])
  # mae = mean_absolute_error(y_true=test_df['y'], y_pred=test_forecast['yhat'])
  # mse = mean_squared_error(y_true=test_df['y'], y_pred=test_forecast['yhat'])
  # rmse = sqrt(mse)
  # r2 = r2_score(y_true=test_df['y'], y_pred=test_forecast['yhat'])
  
  # print("\nModel Error:")
  # print(f"Mean Absolute Error (MAE): {mae}")
  # print(f"Mean Squared Error (MSE): {mse}")
  # print(f"Root Mean Squared Error (RMSE): {rmse}")
  # print(f"R² Score: {r2}")
  # print(f"Mean Absolute Percentage Error (MAPE): {(mape*100):.2f}%")

  # --------------------------------------------------------------------------------------------- #
  # TEST CODE FUNCTIONS FOR ALL 4069 UNIQUE ROUTES (>150s to run)
  # --------------------------------------------------------------------------------------------- #

  ## Find all Unique Routes and their Longest Subsequence of Dates with Min Count 50
  unique_routes = sorted(set(data['route']))  # ['ABE-CHI', 'ABE-MCO', ...]

  routes_tsd = {}  # empty dict to store time-series DFs from processed csv data
  future_tsd = {}  # empty dict to store time-series DFs from Prophet model
  mape_dict = {}  # empty dict to store mape from Prophet predictions on Train-Test Split

  # Loop through all 4069 unique routes
  for route in unique_routes:
    src, dst = route.split('-')
    filtered_df = filter_routes(data, air1=src, air2=dst, min_rows=50)  # return DF or None
    if filtered_df is None:
      continue
    df = find_longest_timeseq(data=filtered_df, ycol='fare', min_rows=50)  # return DF or None
    if (df is not None):
      routes_tsd[route] = df  # store into dict
      # df, ignore = train_test_split(df, test_split='2020-03-21')  # split df on COVID
      train_df, test_df = train_test_split(df, test_split=8)  # test set is last 8 rows
      m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=2)  # instantiate prophet model
      test_fcst = fbp_predict_test(model=m, train=train_df, test=test_df)  # make forecasts
      mape = mean_absolute_percentage_error(y_true=test_df['y'], y_pred=test_fcst['yhat'])
      mape_dict[route] = mape  # store into dict
      print(f'MAPE: {mape*100:.2f}% ({route})')

  print("\nModel Error for All Unique Routes:")
  print(f"min MAPE: {min(mape_dict.values())*100:.2f}% ({min(mape_dict, key=mape_dict.get)})")
  print(f"max MAPE: {max(mape_dict.values())*100:.2f}% ({max(mape_dict, key=mape_dict.get)})")
  print(f"avg MAPE: {sum(mape_dict.values())/len(mape_dict)*100:.2f}%")

  ## Flight Data Information
  # print(len(unique_routes)) # 4069

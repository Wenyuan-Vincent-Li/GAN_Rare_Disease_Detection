"""
This is a utils python file that used by Data.
"""
import math
import datetime
import pandas as pd
import numpy as np

def TimeDiff(first_dt, last_dt):
  # Caculate the time difference between (first_dt and last_dt) in day
  if pd.isnull(first_dt) or pd.isnull(last_dt):
    return np.nan
  datetime_1 = float_to_datetime(first_dt)
  datetime_2 = float_to_datetime(last_dt)
  return (datetime_2 - datetime_1).days

def float_to_datetime(element):
  # Convert the float elemnt to datetime object
  # Ex: 3222012 means 3/22/2012 (mm/dd/yy)
  element = int(element)
  year = element - math.floor(element / 10000) * 10000
  element = math.floor(element / 10000)
  date = element % 100
  month = math.floor(element / 100)
  return datetime.date(year, month, date)

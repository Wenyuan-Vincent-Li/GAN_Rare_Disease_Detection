"""
This is a python file that do the preprocessing of the data (normalization).
"""
## Import module
import pandas as pd
import numpy as np
from sklearn import preprocessing

## Read in the csv file
file_path = 'Data/Data_File/AFT_FS_DT_All_BF_Norm.csv'

data = \
    pd.read_csv(file_path) # data shape [1794919, 1330] (the 1st one of 1330 is Unnamed) 

data_copy = data.copy()

## Split the train and test data
train = data_copy.loc[data_copy.HOLDOUT == 1]
test = data_copy.loc[data_copy.HOLDOUT == 0]

## Drop the Holdout column
train = train.drop(['HOLDOUT'], axis = 1)
test = test.drop(['HOLDOUT'], axis = 1)

## Normalization
method = "Min_Max" # Choose the normalization method in ["Gaussian", "Min_Max"].
columns = [x for x in range(train.columns.get_loc("PAT_AGE"), len(train.columns))] # List the feature column
if method == "Gaussian":
  ## Gausiaan distribution
  mean = train.iloc[:, columns].mean()
  std = train.iloc[:, columns].std() + 1e-12 # for computation stability
  train.iloc[:, columns] = (train.iloc[:, columns] - mean) / std
  test.iloc[:, columns] = (test.iloc[:, columns] - mean) / std
elif method == "Min_Max":
  ## Min_Max normaliztion
  min_max_scaler = preprocessing.MinMaxScaler()
  train.iloc[:, columns] = 2 * min_max_scaler.fit_transform(train.iloc[:, columns]) - 1
  test.iloc[:, columns] = 2 * min_max_scaler.fit_transform(test.iloc[:, columns]) - 1

## Save the train and test dataset, rename the saving path
train.to_csv('Data/Data_File/Train_AFT_FS_All_MinMax.csv', index = False)
test.to_csv('Data/Data_File/Test_AFT_FS_All_MinMax.csv', index = False)
"""
This is a python file that used to explore the EPI data.
"""
## Import module
import pandas as pd
import numpy as np

## Read in the csv file
file_path = 'Data/Data_File/Positive_Negative_Patient_Medical_Hist_File_V3.csv'

data = \
    pd.read_csv(file_path)

## Print the shape of the data  
print(len(data.index), len(data.columns)) # $ patient numer & patient feature

## View the first three data
data.head(3)

## View the columns 
data.columns

for i in range(0, len(data.columns), 5):
  print(data.iloc[0 : 3, i : i + 5])

data_1 = data.iloc[0:10, 970:975]
data.columns.get_loc("_ht3_antag_flag")

## Rename the lower_case symptom
data.rename(columns={'_ht3_antag_flag': '_HT_3_ANTAG_FLAG', 
                       '_ht3_antag_first_exp_dt': '_HT_3_ANTAG_FIRST_EXP_DT',
                       '_ht3_antag_last_exp_dt': '_HT_3_ANTAG_LAST_EXP_DT',
                       '_ht3_antag_count': '_HT_3_ANTAG_COUNT',
                       '_ht3_antag_freq': '_HT_3_ANTAG_FREQ'},inplace=True)

## Select the useful feature
feature_columns = [0, 1] + [x for x in range(18, len(data.columns))]

data_select = data.iloc[:, feature_columns]
data_select.to_csv('Data/Data_File/ALL_DATA_w_FeatureSelection.csv', index = False)  

## View the columns types
data_select.dtypes


## Create a small dataset for data manipulation validation
df1 = data_select.iloc[0:20, 0:9]
df1.dropna(how='any')
df1.to_csv('Data/Data_File/Smaller_Dataset.csv', index = False)

df2 = data_select.iloc[0:20, 0:14]
df2.to_csv('Data/Data_File/Smaller_Dataset_w_2symptom.csv', index = False)

df3 = data_select.iloc[0:10, :]
df3.to_csv('Data/Data_File/Smaller_Dataset_w_allsymptom.csv', index = False)

df4 = data_select.iloc[0:20, 0:29]
df4.to_csv('Data/Data_File/Smaller_Dataset_w_5symptom.csv', index = False)
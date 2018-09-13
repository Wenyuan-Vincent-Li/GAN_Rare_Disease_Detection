"""
This is a python file that calculate the time diffirence of the each symptom and save
it to csv.
"""
## Import module
import pandas as pd
import numpy as np
from sklearn import preprocessing

from Data.utils import TimeDiff # function that used to calculate the time difference in day

## Read in the csv file
file_path = 'Data/Data_File/ALL_DATA_w_FeatureSelection.csv'

data = \
    pd.read_csv(file_path) # data shape [1794919, 1330] (the 1st one of 1330 is Unnamed) 

## View data and data type
data
data.dtypes

data_copy = data.copy()

## Convert pat_gender: F->-1 M->1
data_copy.loc[data_copy.PAT_GENDER == 'F', 'PAT_GENDER'] = -1
data_copy.loc[data_copy.PAT_GENDER == 'M', 'PAT_GENDER'] = 1
data_copy.PAT_GENDER = data_copy.PAT_GENDER.astype(int)

## Caculate the claim time difference for each symptom
## Here we used a range by range approach (specified by symtom_colum), 
## as the kernel alwasy be deat after a while.

symptom_list = []
num_symptom = 0
df_whole = []
symtom_colum = [1000, 1330]

for idx, x in enumerate(data_copy.columns):
  if x.endswith('_FLAG') and (symtom_colum[0] < idx <=  symtom_colum[1]):
    
    symptom = x.split('_FLAG')[0]
    
    EXP_DT = data_copy.loc[:, [symptom +'_FIRST_EXP_DT', symptom +'_LAST_EXP_DT']]

    DT = EXP_DT.apply(lambda x: TimeDiff(x[symptom +'_FIRST_EXP_DT'], \
                                             x[symptom +'_LAST_EXP_DT']), axis=1)
    df = pd.DataFrame({symptom + '_EXP_TD': DT})
    df_whole.append(df)
    symptom_list.append(symptom)
    num_symptom +=1
    print(idx, symptom)

# Concatenate all the data and save it to csv
result = pd.concat(df_whole, axis = 1)
result.to_csv('Data/Data_File/AFT_FS_DT_Disease_%d_2_%d.csv'%(symtom_colum[0], symtom_colum[1]), \
              index = False)
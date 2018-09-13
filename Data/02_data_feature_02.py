"""
This is a python file that inserts the Time Difference results after each symptom
and drops unrelated patient features.
"""
## Import module
import pandas as pd
import numpy as np

## Read the file that contains the calculated Time Difference data
file_path_DT = 'Data/Data_File/AFT_FS_DT.csv'
DT_data = \
    pd.read_csv(file_path_DT)
DT_data_copy = DT_data.copy()
max_DT = DT_data_copy.max(axis = 0)
DT_data_copy = DT_data_copy.fillna(value = max_DT)

## Read the file that contains the featrues
file_path_org = 'Data/Data_File/ALL_DATA_w_FeatureSelection.csv'
ORG_data = pd.read_csv(file_path_org)
ORG_data_copy = ORG_data.copy()

## Insert the TD data after each symptom
num_symptom = 0
symptom_list = []
for idx, x in enumerate(ORG_data_copy.columns):
  if x.endswith('_FLAG'):
    symptom = x.split('_FLAG')[0]
    ORG_data_copy.insert(idx + 5 + num_symptom, symptom + '_EXP_TD', DT_data_copy[symptom + '_EXP_TD'])
    symptom_list.append(symptom)
    num_symptom +=1
    print(idx, symptom)

print("There are %d different symptom in total!"%num_symptom)
    
## Drop the unrelated feature of this symptom
drop_list = []
for symptom in symptom_list:
  drop_list += [symptom +'_FIRST_EXP_DT', symptom +'_LAST_EXP_DT',\
                               symptom + '_FLAG']
ORG_data_copy = ORG_data_copy.drop(drop_list, axis = 1)

## Convert pat_gender: F->-1 M->1
ORG_data_copy.loc[ORG_data_copy.PAT_GENDER == 'F', 'PAT_GENDER'] = -1
ORG_data_copy.loc[ORG_data_copy.PAT_GENDER == 'M', 'PAT_GENDER'] = 1
ORG_data_copy.PAT_GENDER = ORG_data_copy.PAT_GENDER.astype(int)

## Set missing value
ORG_data_copy = ORG_data_copy.fillna(value = 0)

## Save data
ORG_data_copy.to_csv('Data/Data_File/AFT_FS_DT_All_BF_Norm.csv', index = False)
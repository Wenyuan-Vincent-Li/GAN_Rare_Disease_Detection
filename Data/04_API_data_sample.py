import pandas as pd

## Read in the csv file
file_path = 'Data/Data_File/Test_AFT_FS_All_MinMax_SHUFFLE.csv'
data = \
    pd.read_csv(file_path) # data shape [604396, 798] (the 1st one of 1330 is Unnamed)
data_copy = data.copy()

## Generate sample data
pos_data = data_copy.loc[data_copy.OUTCOME == 1].sample(n = 20)
neg_data = data_copy.loc[data_copy.OUTCOME == 0].sample(n = 20)
sample_data = pd.concat([pos_data, neg_data])

for i in range(5):
  sample_data = sample_data.sample(frac = 1)
sample_data.to_csv('Data/Data_File/API_SERVER_SAMPLE_5.csv', \
                   index = False)

## Test in sample data
file_path = 'Data/Data_File/API_SERVER_SAMPLE_5.csv'
data_api = pd.read_csv(file_path)
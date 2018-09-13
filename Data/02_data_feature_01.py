"""
This is a simple python file combine all the Time Difference results together.
"""
## Import module
import pandas as pd
import numpy as np

file_path_1 = 'Data/Data_File/AFT_FS_DT_Disease_0_2_500.csv'
file_path_2 = 'Data/Data_File/AFT_FS_DT_Disease_500_2_1000.csv'
file_path_3 = 'Data/Data_File/AFT_FS_DT_Disease_1000_2_1330.csv'

data_1 = \
    pd.read_csv(file_path_1)
data_2 = \
    pd.read_csv(file_path_2)
data_3 = \
    pd.read_csv(file_path_3)

combine = pd.concat([data_1, data_2, data_3], axis = 1)
combine.to_csv('Data/Data_File/AFT_FS_DT.csv', index = False)
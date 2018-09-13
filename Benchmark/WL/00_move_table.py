"""
Move below 2 tables to hdfs
  - Local location: Data/Data_File/
  - File name: 
      Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv
      Test_AFT_FS_All_MinMax_SHUFFLE.csv
  - hdfs location: /user/fzhang1/rare_disease/

"""

!hadoop fs -mkdir -p /user/fzhang1/rare_disease/

!hadoop fs -copyFromLocal \
  Data/Data_File/Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv \
  /user/fzhang1/rare_disease/
    
!hadoop fs -copyFromLocal \
  Data/Data_File/Test_AFT_FS_All_MinMax_SHUFFLE.csv \
  /user/fzhang1/rare_disease/
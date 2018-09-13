
print("Transfer decrypted files to hdfs")
!hadoop fs -copyFromLocal -f Data/Data_File/Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv /user/yunlong.wang/Abbvie_EPI/temp/
!hadoop fs -copyFromLocal -f Data/Data_File/Test_AFT_FS_All_MinMax_ SHUFFLE.csv /user/yunlong.wang/Abbvie_EPI/temp/
print("Completed transfer")




HDFS_PATH = '/user/yunlong.wang/Abbvie_EPI/temp'
data_train = spark.read.csv('{}/Train_AFT_FS_All.csv'.format(HDFS_PATH), inferSchema=True, header=True)
data_test = spark.read.csv('{}/Test_AFT_FS_All.csv'.format(HDFS_PATH), inferSchema=True, header=True)




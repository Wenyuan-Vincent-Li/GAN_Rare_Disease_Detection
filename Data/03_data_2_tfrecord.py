"""
This is a python file that convert the patient record data to tfrecord.
"""
## Import module
import pandas as pd
import numpy as np
import tensorflow as tf
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  
def csv_2_TF(csv_data, tf_filename):
  '''
  A function that conver csv_data to tfrecord.
  Note that the csv_data is reqired to have the format:
  label, feature 0, feature 1, ... , feature n.
  '''
  with tf.python_io.TFRecordWriter(tf_filename) as writer:
    for row in csv_data:
      # file format : label, feature 0, feature 1, ... , feature n.
      label, features = int(row[0]), row[1:].tolist()
      example = tf.train.Example(features=tf.train.Features(
                feature={
                    'attribute': _float_feature(features),
                    'label': _int64_feature(label)
                    }))
      writer.write(example.SerializeToString())

    
def main_generate_tfrecords():
  ## Read in the data
  train_filename = "Data/Data_File/Train_AFT_FS_All_MinMax.csv"
  test_filename = "Data/Data_File/Test_AFT_FS_All_MinMax.csv"
  train_data = pd.read_csv(train_filename)
  test_data = pd.read_csv(test_filename)
  
  train_data_copy = train_data.copy()
  test_data_copy = test_data.copy()
  
  print("Finishing reading data!")
  
  ## Make a flag named 'Lab_Used_In_Train' that indicates which samples are used 
  ## as labeld data in training.
  train_data.insert(1, 'Lab_Used_In_Train', 0)
  
  ## Fetch the pos_data.
  pos_data = train_data.loc[train_data.OUTCOME == 1]
  
  ## Sample the negative data with 3 times of the pos_data size.
  num_pos = pos_data.shape[0]
  vir_neg = train_data.loc[train_data.OUTCOME == 0].sample(n = 3 * num_pos, replace=False)
  vir_neg.OUTCOME = -1
  
  ## Label the data used in training and update the whole dataset
  vir_neg.Lab_Used_In_Train = 1
  pos_data.Lab_Used_In_Train = 1
  train_data.update(vir_neg)  
  train_data.update(pos_data)
  
  ## Fetch the data that not used as labeld data in training
  unl_data = train_data.loc[train_data.OUTCOME == 0]
  
  ## Convert back the ver_neg.OUTCOME
  vir_neg.OUTCOME = 0
  train_data.update(vir_neg)
  
  ## Create the labeld data used in training, and combine the unlabeld data with test data
  ## for validation purpose.
  lab_data = pd.concat([pos_data, vir_neg])
  test_data_val_unl = pd.concat([test_data, unl_data.drop('Lab_Used_In_Train', axis = 1)])
  
  ## Shuffle the data 5 times to make it random order.
  for i in range(5):
    train_data = train_data.sample(frac = 1) # training data have "lab_used_in_train"
    lab_data = lab_data.sample(frac = 1) # label data
    unl_data = unl_data.sample(frac = 1) # unlabel data
    test_data = test_data.sample(frac = 1) # test data
    test_data_val_unl = test_data_val_unl.sample(frac = 1) # test combine data
  
  ## Save the SHUFFLE data to csv with 'Lab_Used_In_Train' for bechmark modeling.
  train_data.to_csv('Data/Data_File/Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv', index = False)
  test_data.to_csv('Data/Data_File/Test_AFT_FS_All_MinMax_SHUFFLE.csv', index = False)
  
  ## Drop the unnecessary column for tfrecord
  lab_data = lab_data.drop('Lab_Used_In_Train', axis = 1)
  unl_data = unl_data.drop('Lab_Used_In_Train', axis = 1)
  ## Convert the data into tfrecord
  lab_data = lab_data.values
  unl_data = unl_data.values
  test_data_val_unl = test_data_val_unl.values
  
  ## Convert the data to tfrecord and save the data
  tf_filename_lab = os.path.splitext(train_filename)[0] + '_Lab' + '.tfrecords'
  tf_filename_unl = os.path.splitext(train_filename)[0] + '_Unl' + '.tfrecords'
  tf_filename_test = os.path.splitext(test_filename)[0] + '_Val_Unl' + '.tfrecords'
  csv_2_TF(lab_data, tf_filename_lab)
  csv_2_TF(unl_data, tf_filename_unl)
  csv_2_TF(test_data_val_unl, tf_filename_test)

if __name__ == "__main__":
  '''
  After the conversion, the data statistics are:
  
  Training_data: [1190523, 798]:
      Lab_data: [23692, 798]:
          Pos_data: [5923, 798]
          Neg_daat: [17769, 798]
      Unl_data: [1166831, 798]
  
  Testing_data: [1771227, 798]:
      Pos_data: [23246, 798]
      Neg_data: [1747981, 798]
      
  Three files are used for training and testing in GAN:
  Train_AFT_FS_All_MinMax_Lab.tfrecords
  Train_AFT_FS_All_MinMax_Unl.tfrecords
  Test_AFT_FS_All_MinMax_Val_Unl.tfrecords
  
  Two csv files are used for the bechmark baseline:
  Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv
  Test_AFT_FS_All_MinMax_SHUFFLE.csv
  '''
  main_generate_tfrecords()
  
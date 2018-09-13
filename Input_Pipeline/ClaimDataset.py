"""
This is a python file that create a ClaimDataSet class
"""
## import module
import os
import tensorflow as tf

class ClaimDataSet(object):
  """
  Claim_Level Dataset
  """
  def __init__(self, data_dir, file_mid_name, config = None, subset='Train'):
    self._is_training_input = tf.placeholder(tf.bool, name = 'Is_training_input')
    self.data_dir = data_dir
    self.subset = subset
    self.file_mid_name = file_mid_name
    self.config = config
  
  def get_filenames(self):
    ## Return the tfrecord filename based on ['Train', 'Test'] stage in a list.
    if self.subset in ['Train', 'Test']:
      if self.subset == 'Train':
        return [os.path.join(self.data_dir, self.subset\
                           + self.file_mid_name + '_Lab' + '.tfrecords'),
                os.path.join(self.data_dir, self.subset\
                           + self.file_mid_name + '_Unl' + '.tfrecords')]
      else:
        return [os.path.join(self.data_dir, self.subset \
                          + self.file_mid_name +'_Val_Unl.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)
      
    
  def input_from_tfrecord_filename(self):
    ## Read in datasets according to filename and return a object list
    dataset = []
    filename = self.get_filenames()
    for name in filename:
      dataset.append(tf.data.TFRecordDataset(name))
    return dataset
  
  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'attribute': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'label': tf.FixedLenFeature([], tf.int64),
        })
        
    feature = tf.cast(features['attribute'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
        
    ## pre-processing data, make one-hot encoding
    feature, label = self.pre_processing(feature, label)
    return feature, label
  
  def pre_processing(self, feature, label):
    ## Make one-hot label encoding
    ## Add other pre-processing methods that you think are necessary
    label = tf.one_hot(label, self.config.NUM_CLASSES)
    return [feature, label]
  
  def shuffle_and_repeat(self, dataset, repeat = 1):
    dataset = dataset.shuffle(buffer_size = \
              self.config.MIN_QUEUE_EXAMPLES + \
              30 * self.config.BATCH_SIZE, \
              )
    dataset = dataset.repeat(count = repeat)    
    return dataset
    
  def batch(self, dataset, batch_ratio = 1):
    dataset = dataset.batch(batch_size = batch_ratio * self.config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size= 30 * self.config.BATCH_SIZE)
    return dataset
  
  def inputpipline_test(self):
    ## Inputpipline that used for validation
    ## Return: init_op (list); val_input (tensor); val_lab (tensor).
    
    # 1 Read in tfrecords
    dataset_val = self.input_from_tfrecord_filename()[0]    
    # 2 Parser tfrecords and preprocessing the data
    dataset_val = dataset_val.map(self.parser, \
                                 num_parallel_calls=self.config.BATCH_SIZE)  
    # 3 Shuffle and repeat
    dataset_val = dataset_val.repeat(count = 1)
    # 4 Batch it up
    dataset_val = self.batch(dataset_val)
    # 5 Make iterator
    iterator = dataset_val.make_initializable_iterator()
    val_input, val_lab = iterator.get_next()
    init_op = [iterator.initializer]
    return init_op, val_input, val_lab
  
  def inputpipline_train_val(self, other):
    ## Inputpipline that used for training
    ## Return: init_op_train (list); init_op_val (list); lab_input (tensor);
    ## lab_output (tensor); train_unl_input (tensor). 
    
    # 1 Read in tfrecords
    dataset_train_lab, dataset_train_unl = self.input_from_tfrecord_filename()
    dataset_val = other.input_from_tfrecord_filename()[0]
    # 2 Parser tfrecords and preprocessing the data
    dataset_train_lab = dataset_train_lab.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
    dataset_train_unl = dataset_train_unl.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
    dataset_val = dataset_val.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
    # 3 Shuffle and repeat
    dataset_train_lab = self.shuffle_and_repeat(dataset_train_lab, repeat = -1)
    dataset_train_unl = self.shuffle_and_repeat(dataset_train_unl, repeat = 1)
    dataset_val = self.shuffle_and_repeat(dataset_val, repeat = -1)    
    # 4 Batch it up
    dataset_train_lab = self.batch(dataset_train_lab)
    dataset_train_unl = self.batch(dataset_train_unl)
    dataset_val = self.batch(dataset_val)    
    # 5 Make iterator
    # first make the labeled data structure iterator
    lab_iterator = tf.data.Iterator.from_structure(dataset_train_lab.output_types,\
                                                   dataset_train_lab.output_shapes)
    lab_input, lab_output = lab_iterator.get_next()
    # then make the unl data iterator
    train_unl_iterator = dataset_train_unl.make_initializable_iterator()
    train_unl_input, _ = train_unl_iterator.get_next()    
    train_unl_input = tf.concat([lab_input, train_unl_input], 0) # make more postive samples in unl
    # finally make initializer
    init_op_train_lab = lab_iterator.make_initializer(dataset_train_lab)
    init_op_val_lab = lab_iterator.make_initializer(dataset_val)
    init_op_train = [init_op_train_lab, train_unl_iterator.initializer]
    init_op_val = [init_op_val_lab]
        
    return init_op_train, init_op_val, \
           lab_input, lab_output, train_unl_input
  
  
def _main_train_val_test():
  import os
  import sys
  sys.path.append('/home/cdsw')
  from Config.config import Config
  import time
  tf.logging.set_verbosity(tf.logging.INFO)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all debugging logs
  
  class Temp_Config(Config):
    BATCH_SIZE = 2048
    VAL_STEP = 100
  tmp_config = Temp_Config()
  
  data_dir = "Data/Data_File"
  file_mid_name = "_AFT_FS_All_MinMax"
  
  tf.reset_default_graph()
  with tf.device('/cpu:0'):
    dataset_train = ClaimDataSet(data_dir, file_mid_name, tmp_config, 'Train')
    dataset_test = ClaimDataSet(data_dir, file_mid_name, tmp_config, 'Test')
  
    init_op_train, init_op_val, feature_batch, label_batch, unl_batch \
          = dataset_train.inputpipline_train_val(dataset_test)
    
    num_unl = 0
    num_batch = 0
    with tf.Session() as sess:
      sess.run(init_op_train)
      while True:
        try:
          feature_batch_o, label_batch_o, unl_batch_o = \
              sess.run([feature_batch, label_batch, unl_batch], \
                       feed_dict={dataset_train._is_training_input: True})
          num_unl += unl_batch_o.shape[0] - tmp_config.BATCH_SIZE
          num_batch += 1
        except tf.errors.OutOfRangeError:
          train_batch_shape = feature_batch_o.shape
          break;
      sess.run(init_op_val)
      for _ in range(tmp_config.VAL_STEP):
        feature_batch_o, label_batch_o = \
              sess.run([feature_batch, label_batch], \
                       feed_dict={dataset_train._is_training_input: False})
        val_batch_shape = feature_batch_o.shape
    
    ## Print the Statistics
    print("TRAN_VAL STATISTICS: \n",\
         "NUM_UNL_DATA: %d \n"%num_unl,\
         "NUM_BATCH_PER_EPOCH: %d \n"%num_batch,\
         "BATCH_SIZE_TRAIN: ", train_batch_shape, '\n',\
         "BATCH_SIZE_VAL:", val_batch_shape)
    
    
def _main_val_test():
  import os
  import sys
  sys.path.append('/home/cdsw')
  from Config.config import Config
  import time
  tf.logging.set_verbosity(tf.logging.INFO)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all debugging logs
  
  class Temp_Config(Config):
    BATCH_SIZE = 4096
  tmp_config = Temp_Config()
  
  data_dir = "Data/Data_File"
  file_mid_name = "_AFT_FS_All_MinMax"
  
  tf.reset_default_graph()
  with tf.device('/cpu:0'):
    dataset_test = ClaimDataSet(data_dir, file_mid_name, tmp_config, 'Test')
  
    init_op, val_input, val_lab \
          = dataset_test.inputpipline_test()
    val_lab = tf.argmax(val_lab, axis = -1)
    num_test = 0
    num_pos = 0
    batch_size_val = None
    with tf.Session() as sess:
      sess.run(init_op)
      while True:
        try:        
          val_input_o, val_lab_o = \
                  sess.run([val_input, val_lab], feed_dict={dataset_test._is_training_input: False})
          num_test += val_input_o.shape[0]
          num_pos += sum(val_lab_o)
          if batch_size_val == None:
            batch_size_val = val_input_o.shape
        except tf.errors.OutOfRangeError:
          break
  ## Print the Statistics
  print("TEST STATISTICS: \n",\
         "NUM_TEST_DATA: %d \n"%num_test,\
         "BATCH_SIZE_TEST:", batch_size_val, '\n',\
         "NUM_POS_DATA:%d"%num_pos)
    
  
if __name__ == "__main__":
  ## Test the trainig stage
  _main_train_val_test()
  ## Test the evaluation stage
  _main_val_test()
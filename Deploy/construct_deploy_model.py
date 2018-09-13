'''
This is a python file that used for constructing a final model for API serving.
For example, here we trained a GAN model that leverage huge amount of un-labeled data.
However, in the real application, we only need the discriminator. So we discard generator,
construct the discriminator and only restore the weights for it.
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from Training.Saver import Saver
from Testing.utils import initialize_uninitialized_vars

class Deploy_Model(object):
  def __init__(self, config, save_dir):
    self.config = config
    self.save_dir = save_dir
    
  def deploy_model(self, model, dir_names = None, epoch = None):
    # Reset the tensorflow graph
    tf.reset_default_graph()
    
    # Build up the graph
    with tf.device('/gpu:0'):
      input_feature, logits, preds, probs, main_graph\
        = self._build_inference_graph(model) 
    
    # Add saver
    saver = Saver(self.save_dir)
    
    # Prepare the input array
    self._input_processing()
    
    # Create a session
    with tf.Session() as sess:
      # restore the weights
      _ = saver.restore(sess, dir_names = dir_names, epoch = epoch)
      # initialize the unitialized variables
      initialize_uninitialized_vars(sess)
      logits_o, probs_o, preds_o = sess.run([logits, preds, probs], \
                                            feed_dict = {input_feature: self.input_array,
                                                        main_graph.is_training: False})
      saver.save(sess, 'model_final' + '.ckpt')
      # Note this save the model in the original folder, you need manully move the file
      # to Deploy/FinalModel
      # TODO: save the model directly in Deploy/FinalModel
    return logits_o, probs_o, preds_o
      
  def _input_processing(self):
    '''
    Function used to read in data and do pre-processing.
    '''
    file_path = 'Data/Data_File/API_SERVER_SAMPLE.csv'
    data_api = pd.read_csv(file_path)
    data_api = data_api.drop('OUTCOME', axis = 1)
    self.input_array = data_api.values
    
  
  def _build_inference_graph(self, model):
    '''
    Function used to create the inference graph
    '''
    main_graph = model(self.config)
    # Create a placeholder for the input
    input_feature = tf.placeholder(tf.float32, [None, self.config.FEATURE_NUM], \
                             name = "input_feature")
    # Construct the main graph
    logits = main_graph.forward_pass_for_test(input_feature)
    with tf.name_scope('Results'):
      preds = tf.argmax(logits, axis = -1)
      probs = tf.nn.softmax(logits)[:, 1]
    
    # Add input and output to tf collection
    tf.add_to_collection("input_feature", input_feature)
    tf.add_to_collection("logits", logits)
    tf.add_to_collection("probs", probs)
    tf.add_to_collection("preds", preds)
    
    return input_feature, logits, probs, preds, main_graph

if __name__ == "__main__":
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all debugging logs
  tf.logging.set_verbosity(tf.logging.INFO)
  
  from Config.config import Config
  from Model.GAN_00 import GAN_00 as Model

  class TempConfig(Config):
    SUMMARY = False
  
  # Create a global configuration object
  tmp_config = TempConfig()
  
  ## Specify the trained weights localtion
  save_dir = "Training/Weight_MinMax_GAN_Val_Unl" # Folder that saves the trained weights
  # Specify the Run, choose either Run = None (find the latest run) 
  Run = ['Run_2018-08-03_15_10_34']
  epoch = 4 # Specify the epoch
  
  # Create a evaler object
  Deploy = Deploy_Model(tmp_config, save_dir)
  # Run evaluation
  logits, probs, preds = Deploy.deploy_model(Model, dir_names = Run, epoch = epoch)
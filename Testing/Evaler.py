'''
This is a python file that used for evaluate GAN.
'''
import tensorflow as tf
from Input_Pipeline.ClaimDataset import ClaimDataSet as DataSet
from Training.Saver import Saver
from Testing.eval_base import Evaler_base
from Testing.utils import save_dict_as_txt, initialize_uninitialized_vars, \
                          convert_list_2_nparray

class Evaler(Evaler_base):
  def __init__(self, config, save_dir):
    self.config = config
    self.save_dir = save_dir
    
  def evaler(self, model, dir_names = None, epoch = None):
    # Reset the tensorflow graph
    tf.reset_default_graph()
    # Input node
    init_val_op, val_input, val_lab = self._input_fn_eval()
    val_lab = tf.argmax(val_lab, axis = -1)
    # Build up the graph
    with tf.device('/gpu:0'):
      logits, preds, probs, accuracy, update_op_a, roc_auc, update_op_roc,\
      pr_auc, update_op_pr, main_graph\
        = self._build_test_graph(val_input, val_lab, model)    
    
    # Add saver
    saver = Saver(self.save_dir)
    
    # List to store the results
    Val_lab = []
    Preds = []
    Logits = []
    Probs = []
    
    # Create a session
    with tf.Session() as sess:
      # restore the weights
      _ = saver.restore(sess, dir_names = dir_names, epoch = epoch)
      # initialize the unitialized variables
      initialize_uninitialized_vars(sess)
      # initialize the dataset iterator
      sess.run(init_val_op)
      # start evaluation
      count = 1
      while True:
        try:
          val_lab_o, logits_o, preds_o, probs_o, accuracy_o, roc_auc_o, pr_auc_o,\
          _, _, _ = \
            sess.run([val_lab, logits, preds, probs, accuracy, \
                      roc_auc, pr_auc, update_op_a, update_op_roc, update_op_pr], \
                      feed_dict={main_graph.is_training: False})
          # store results
          Val_lab.append(val_lab_o)
          Preds.append(preds_o)
          Logits.append(logits_o)
          Probs.append(probs_o[:, -1])
          tf.logging.debug("The current validation sample batch num is {}."\
                              .format(count))
          count += 1
        except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
          # print out the evaluation results
          tf.logging.info("The validation results are: accuracy {:.2f}; roc_auc {:.2f}; pr_auc {:.2f}."\
                              .format(accuracy_o, roc_auc_o, pr_auc_o))
          break
    return Val_lab, Preds, Logits, Probs, accuracy_o, roc_auc_o, pr_auc_o
      
  def _input_fn_eval(self):
    '''
    Function used to create the input node.
    '''
    with tf.device('/cpu:0'):
      with tf.name_scope('Input_Data'):
        dataset_val = DataSet(self.config.DATA_DIR, self.config.FILE_MID_NAME,\
                             self.config, 'Test')
        iterator, val_input, val_lab = dataset_val.inputpipline_test()
        
        val_input.set_shape([None, \
                            self.config.FEATURE_NUM])
    return iterator, val_input, val_lab
  
  def _build_test_graph(self, val_input, val_lab, model):
    '''
    Function used to create the eval graph
    '''
    main_graph = model(self.config)
    logits = main_graph.forward_pass_for_test(val_input)
    with tf.name_scope('Results'):
      preds = tf.argmax(logits, axis = -1)
      probs = tf.nn.softmax(logits)
    
    # create metric
    accuracy, update_op_a, roc_auc, update_op_roc, pr_auc, update_op_pr = \
      self._metric(val_lab, preds, probs[:, -1])

    return logits, preds, probs, accuracy, update_op_a,\
            roc_auc, update_op_roc, pr_auc, update_op_pr, main_graph
  
  def _metric(self, real_lab, prediction, probs):
    '''
    Function used to create evaluation metric
    '''
    with tf.name_scope('Metric'):
      accuracy, update_op_a = self._accuracy_metric(real_lab, prediction)
      with tf.device('/cpu:0'):
        roc_auc, update_op_roc = self._auc_metric(real_lab, probs, curve = 'ROC')
        pr_auc, update_op_pr = self._auc_metric(real_lab, probs, curve = 'PR')
    return accuracy, update_op_a, roc_auc, update_op_roc, pr_auc, update_op_pr
  
if __name__ == "__main__":
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all debugging logs
  tf.logging.set_verbosity(tf.logging.INFO)
  
  from Config.config import Config
  from Model.GAN_00 import GAN_00 as Model
  from Testing.Analysis_00 import Analysis

  class TempConfig(Config):
    DATA_DIR = "Data/Data_File"
    FILE_MID_NAME = "_AFT_FS_All_MinMax"
    BATCH_SIZE = 10240
    SUMMARY = False
    SAVE = False
  # Create a global configuration object
  tmp_config = TempConfig()
  
  ## Specify the trained weights localtion
  save_dir = "Training/Weight_MinMax_GAN_Val_Unl" # Folder that saves the trained weights
  # Specify the Run, choose either Run = None (find the latest run) 
  # or use Run = ['Run_2018-08-03_15_10_34']
  Run = ['Run_2018-08-03_15_10_34']
  # Run = None
  epoch = 4 # Specify the epoch
  
  # Create a evaler object
  Eval = Evaler(tmp_config, save_dir)
  # Run evaluation
  val_lab, preds, logits, \
  probs, acc, roc, pr = Eval.evaler(Model, dir_names = Run, epoch = epoch)
  
  # Post Analysis
  # 1. Convert all the list to nparray
  val_lab = convert_list_2_nparray(val_lab)
  preds = convert_list_2_nparray(preds)
  probs = convert_list_2_nparray(probs)
  Analysis(val_lab, preds, probs)
  # 2. Save the results
  if tmp_config.SAVE:
    res = {'Val_lab': val_lab, 'Preds': preds, 'Probs': probs}
    dir_name = 'Testing/Results/epoch=' + str(epoch) + '/'
    save_dict_as_txt(res, dir_name) 
  
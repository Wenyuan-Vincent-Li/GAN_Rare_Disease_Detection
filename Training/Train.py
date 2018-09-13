'''
This is a python file that used for training GAN.
TODO: provide a parser access from terminal.
'''
## Import module
import sys
import os
sys.path.append('/home/cdsw')
import numpy as np
import tensorflow as tf
from time import strftime
from datetime import datetime
from pytz import timezone

from Input_Pipeline.ClaimDataset import ClaimDataSet as DataSet
from Training.train_base import Train_base
from Training.Saver import Saver
from Training.Summary import Summary
import Training.utils as utils
from Training.utils import initialize_uninitialized_vars


class Train(Train_base):
  def __init__(self, config, log_dir, save_dir, **kwargs):
    super(Train, self).__init__(config.LEARNING_RATE, config.MOMENTUM)
    self.config = config
    self.save_dir = save_dir
    self.comments = kwargs.get('comments', '')
    if self.config.SUMMARY:
      if self.config.SUMMARY_TRAIN_VAL:
        self.summary_train = Summary(log_dir, config, log_type = 'train', \
                              log_comments = kwargs.get('comments', ''))
        self.summary_val = Summary(log_dir, config, log_type = 'val', \
                              log_comments = kwargs.get('comments', ''))
      else:
        self.summary = Summary(log_dir, config, \
                              log_comments = kwargs.get('comments', ''))
  
  def train(self, model):
    # Reset tf graph.
    tf.reset_default_graph()
    
    # Create input node
    init_op_train, init_op_val, real_lab_input,\
           real_lab, real_unl_input, dataset_train = self._input_fn_train_val()
      
    # Build up the graph
    with tf.device('/gpu:0'):
      d_loss, g_loss, accuracy, roc_auc, pr_auc, \
      update_op, reset_op, preds, probs, main_graph, scalar_train_sum_dict\
                                      = training._build_train_graph(real_lab_input, \
                                      real_unl_input, real_lab, model)
    # Create optimizer
    with tf.name_scope('Train'):
      theta_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Generator')
      theta_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Discriminator')
      optimizer = self._Adam_Optimizer()
      d_solver, d_grads = self._train_op_w_grads(optimizer, d_loss,\
                                            var_list = theta_D)
      g_solver, g_grads = self._train_op_w_grads(optimizer, g_loss,\
                                            var_list = theta_G)
    
    # Print out the variable name in debug mode
    tf.logging.debug(utils.variable_name_string())
    
    # Add summary
    if self.config.SUMMARY:
      if self.config.SUMMARY_TRAIN_VAL:                
        summary_dict_train = {}
        summary_dict_val = {}
        if self.config.SUMMARY_SCALAR:
          scalar_train = {'generator_loss': g_loss, \
                          'discriminator_loss': d_loss}
          scalar_train.update(scalar_train_sum_dict)
          scalar_val = {'val_accuracy': accuracy, \
                        'val_pr_auc': pr_auc, \
                        'val_roc_auc': roc_auc}
          
          summary_dict_train['scalar'] = scalar_train
          summary_dict_val['scalar'] = scalar_val
                
        if self.config.SUMMARY_HISTOGRAM:
          ## TODO: add any vectors that you want to visulize.
          pass
        # Merge summary
        merged_summary_train = \
          self.summary_train.add_summary(summary_dict_train)
        merged_summary_val = \
          self.summary_val.add_summary(summary_dict_val)
                
    # Add saver
    saver = Saver(self.save_dir)
    
    # Whether to use a pre-trained weights 
    if self.config.PRE_TRAIN:
      pre_train_saver = tf.train.Saver(theta_D)
    
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # Use soft_placement to place those variables, which can be placed, on GPU 
    
    # Create Session
    with tf.Session(config = sess_config) as sess:
      # Add graph to tensorboard
      if self.config.SUMMARY and self.config.SUMMARY_GRAPH:
        if self.config.SUMMARY_TRAIN_VAL:
          self.summary_train._graph_summary(sess.graph)
      
      # Restore the weights from the previous training        
      if self.config.RESTORE:
        start_epoch = saver.restore(sess)
      else:
        # Create a new folder for saving model
        saver.set_save_path(comments = self.comments)
        start_epoch = 0
        if self.config.PRE_TRAIN:
          # Restore the pre-trained weights for D
          pre_train_saver.restore(sess, self.config.PRE_TRAIN_FILE_PATH)
          initialize_uninitialized_vars(sess)
        else:
          # initialize the variables
          init_var = tf.group(tf.global_variables_initializer(), \
                              tf.local_variables_initializer())
          sess.run(init_var)
      
      # Start Training
      tf.logging.info("Start training!")
      for epoch in range(1, self.config.EPOCHS + 1):
        tf.logging.info("Training for epoch {}.".format(epoch))
        train_pr_bar = tf.contrib.keras.utils.Progbar(target = \
                          int(tmp_config.TRAIN_SIZE / tmp_config.BATCH_SIZE))
        sess.run(init_op_train, feed_dict = {dataset_train._is_training_input: True})
        
        for i in range(int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE)):
          # Update discriminator
          d_loss_o, _, summary_out  = sess.run([d_loss, d_solver, \
                                   merged_summary_train], \
                                   feed_dict={main_graph.is_training: True, \
                                   dataset_train._is_training_input: True})
          # Update generator
          g_loss_o, _, summary_out = sess.run([g_loss, g_solver, \
                                    merged_summary_train], \
                                    feed_dict={main_graph.is_training: True, \
                                    dataset_train._is_training_input: True})
          
          tf.logging.debug("Training for batch {}.".format(i))
          # Update progress bar
          train_pr_bar.update(i)
        
        if self.config.SUMMARY_TRAIN_VAL:
          # Add summary
          self.summary_train.summary_writer.add_summary(summary_out, epoch + start_epoch)
        
        # Perform validation
        tf.logging.info("\nValidate for epoch {}.".format(epoch))
        sess.run(init_op_val + [reset_op], \
                 feed_dict = {dataset_train._is_training_input: False})
        count = 1
        val_pr_bar = tf.contrib.keras.utils.Progbar(target = \
                                      int(tmp_config.VAL_SIZE / tmp_config.BATCH_SIZE))
        
        for i in range(int(self.config.VAL_SIZE / self.config.BATCH_SIZE)):
          try:
            accuracy_o, summary_out, roc_auc_o, \
            pr_auc_o, val_lab_o, preds_o, \
            probs_o, _, _, _ = sess.run([accuracy, merged_summary_val,\
                                         roc_auc, pr_auc, real_lab, \
                                         preds, probs] + update_op,\
                                         feed_dict={main_graph.is_training: False,\
                                         dataset_train._is_training_input: False})
            
            tf.logging.debug("Validate for batch {}.".format(count))
            # Update progress bar
            val_pr_bar.update(count)
            count += 1 
          except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
            break
        
        tf.logging.info("\nThe current validation accuracy for epoch {} is {:.2f}, roc_auc is {:.2f}, pr_auc is {:.2f}.\n"\
                          .format(epoch, accuracy_o, roc_auc_o, pr_auc_o))
        
        # Add summary to tensorboard
        if self.config.SUMMARY_TRAIN_VAL:
          self.summary_val.summary_writer.add_summary(summary_out, epoch + start_epoch)
        
        # Save the model per SAVE_PER_EPOCH
        if epoch % self.config.SAVE_PER_EPOCH == 0:
          save_name = str(epoch + start_epoch)
          saver.save(sess, 'model_' + save_name.zfill(4) \
                               + '.ckpt')
      
      if self.config.SUMMARY_TRAIN_VAL:
        self.summary_train.summary_writer.flush()
        self.summary_train.summary_writer.close()
        self.summary_val.summary_writer.flush()
        self.summary_val.summary_writer.close()
      
      # Save the model after all epochs
      save_name = str(epoch + start_epoch)
      saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')
    
    return
  
  def _input_fn_train_val(self):
    '''
    Create the input node.
    '''
    with tf.device('/cpu:0'):
      with tf.name_scope('Input_Data'):
        # Training dataset
        dataset_train = DataSet(self.config.DATA_DIR, self.config.FILE_MID_NAME,\
                                self.config, 'Train')
        # Validation dataset
        dataset_val = DataSet(self.config.DATA_DIR, self.config.FILE_MID_NAME,\
                                self.config, 'Test')
        
        # Inputpipeline
        init_op_train_lab, init_op_val_lab,\
            lab_input, lab_output, train_unl_input \
            = dataset_train.inputpipline_train_val(dataset_val)
        
        # Fix the dataset shape for the purpose of dense layer
        lab_input.set_shape([None, \
                            self.config.FEATURE_NUM])
        train_unl_input.set_shape([None, \
                              self.config.FEATURE_NUM])
        tf.add_to_collection("inputs", lab_input)
    return init_op_train_lab, init_op_val_lab, lab_input,\
           lab_output, train_unl_input, dataset_train
    
  def _build_train_graph(self, real_lab_input, real_unl_input, real_lab, model):
    '''
    Build up the training graph.
    '''
    # Create the model
    main_graph = model(self.config)
    real_lab_logits, real_unl_logits, \
    fake_logits, real_unl_feat, \
    real_lab_feat, fake_feat = main_graph.forward_pass(real_lab_input,\
                                                       real_unl_input)
    # Create the loss
    d_loss, g_loss, scaler_train_sum_dict = self._loss(real_lab_logits, \
                      real_unl_logits, fake_logits, real_lab, \
                      real_unl_feat, real_lab_feat, fake_feat)
    # Create the metric
    accuracy, roc_auc, pr_auc, update_op, reset_op, preds, probs = \
        self._metric(real_lab_logits, real_lab)
    
    return d_loss, g_loss, accuracy, roc_auc, pr_auc, update_op, reset_op, \
           preds, probs, main_graph, scaler_train_sum_dict
  
  def _loss(self, d_real_lab_logits, d_real_unl_logits, d_fake_logits, d_real_lab, \
            real_unl_feat, real_lab_feat, fake_feat):
    '''
    Create loss function.
    '''
    with tf.name_scope('Loss'):
      # Use weight in the cross entropy loss as the data is imbalanced.
      weight = tf.where(tf.equal(tf.argmax(d_real_lab, axis = -1), 1), \
                        self.config.BATCH_RATIO * tf.ones_like(\
                        tf.argmax(d_real_lab, axis = -1), dtype = tf.float32),\
                        tf.ones_like(tf.argmax(d_real_lab, axis = -1), \
                                     dtype = tf.float32))
      # Discriminator loss
      # 1 Label data loss
      d_lab = tf.reduce_mean(\
        tf.losses.softmax_cross_entropy(onehot_labels = d_real_lab, \
                                        logits = d_real_lab_logits, \
                                        weights = weight, label_smoothing = 0))
      # 2 Unlabeled data loss
      d_unl = -tf.reduce_mean(tf.reduce_logsumexp(d_real_unl_logits, axis = 1)) + \
                  tf.reduce_mean(tf.nn.softplus(\
                                      tf.reduce_logsumexp(d_real_unl_logits, axis = 1)))
      # 3 Fake data loss
      d_fake = tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(d_fake_logits)))
      # 4 Entropy loss
      if self.config.ENTROPY:
        d_ent = self._entropy(d_real_unl_logits)
      else: 
        d_ent = 0
      # Total discriminator loss
      d_loss = d_lab + d_unl + d_fake + d_ent
      
      # Generator loss
      # 1 Feature mapping loss
      g_fm = tf.reduce_mean(tf.square(tf.reduce_mean(real_unl_feat, axis = 0) \
                                      - tf.reduce_mean(fake_feat, axis = 0)))
      # 2 Pull-away loss    
      if self.config.PULL_AWAY:
        feat_norm = fake_feat / tf.norm(fake_feat, ord='euclidean', axis=1, \
                                        keep_dims=True)
        g_pt1 = tf.tensordot(feat_norm, feat_norm, axes=[[1],[1]])
        g_pt1 = tf.reduce_mean(g_pt1)
        
        feat_norm_lab = real_lab_feat / tf.norm(real_lab_feat, ord='euclidean', \
                                                axis=1, keep_dims=True)
        g_pt2 = tf.tensordot(feat_norm, feat_norm_lab, axes=[[1],[1]])
        g_pt2 = tf.reduce_mean(g_pt2)
        
        g_pt = g_pt1 + g_pt2
      else:
        g_pt = 0
      # Total discriminator loss
      g_loss = g_fm + g_pt
      
      ## Add l2 normalization
      variables = tf.trainable_variables()
      g_vars = [v for v in variables \
                if ('bias' not in v.name) \
                and ('batch_normalization' not in v.name) \
                and ('Generator' in v.name)]
      d_vars = [v for v in variables \
                if ('bias' not in v.name) \
                and ('batch_normalization' not in v.name) \
                and ('Discriminator' in v.name)]
      g_l2 = self._loss_weight_l2(g_vars, eta = self.config.WEIGHT_REG)
      d_l2 = self._loss_weight_l2(d_vars, eta = self.config.WEIGHT_REG)
      
      g_loss += g_l2
      d_loss += d_l2
      
      # Add loss to tensorboard
      scalar_train_sum_dict = {'d_loss_label': d_lab, 'd_loss_unlabel': d_unl,\
                         'd_loss_ent':d_ent, 'g_loss_fm': g_fm, 'g_pt': g_pt}
    
    return d_loss, g_loss, scalar_train_sum_dict  
  
  def _metric(self, real_lab_logits, real_lab):
    '''
    Create evaluation metric.
    '''
    with tf.name_scope('Metric') as scope:
      real_lab = tf.argmax(real_lab, axis = -1)
      prediction = tf.argmax(real_lab_logits, axis = -1)
      probs = tf.nn.softmax(real_lab_logits)[:, -1]
      
      # Pridcition accuracy
      accuracy, update_op_a = self._accuracy_metric(real_lab, prediction)
      
      with tf.device('/cpu:0'):
        # ROC
        roc_auc, update_op_roc = self._auc_metric(real_lab, probs, curve = 'ROC')
        # PR
        pr_auc, update_op_pr = self._auc_metric(real_lab, probs, curve = 'PR')
      
      # Update op inside each validation run
      update_op = [update_op_a, update_op_roc, update_op_pr]
      # Reset op for each validation run
      variables = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
      reset_op = tf.variables_initializer(variables)
      return accuracy, roc_auc, pr_auc, update_op, reset_op, prediction, probs 
        
  
if __name__ == "__main__":
  from Config.config import Config
  from Model.GAN_00 import GAN_00 as Model
  
  tf.logging.set_verbosity(tf.logging.INFO)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs
  
  class TempConfig(Config):
    NAME = "MIN_MAX_GAN"
    
    ## Input pipeline
    DATA_DIR = "Data/Data_File"
    FILE_MID_NAME = "_AFT_FS_All_MinMax"
    BATCH_SIZE = 512
    
    ## Training settings
    # Restore
    RESTORE = False # Whether to use the previous trained weights
    PRE_TRAIN = False # Whether to use the pre-trained weights for discriminator
    PRE_TRAIN_FILE_PATH = None # Specify the pre-trained weights file path
    # Loss settings
    PULL_AWAY = False # Whether to use pull away term
    ENTROPY = False # Whether to use the entropy term
    # Training schedule
    EPOCHS = 10 # Num of epochs to train in the current run
    TRAIN_SIZE = int(1166831 / 10) # Num of samples used to train per epoch
    VAL_SIZE = 10000 # Num of samples used to val after every epoch
    SAVE_PER_EPOCH = 2 # How often to save the trained weights
  
  # Create the global configuration
  tmp_config = TempConfig() 
  # Folder to save the trained weights
  save_dir = "Training/Weight_MinMax_GAN_Val_Unl"
  # Folder to save the tensorboard info
  log_dir = "Training/Log_MinMax_GAN_Val_Unl"
  # Comments log on the current run
  comments = "This training is for creating a developed API."
  comments += tmp_config.config_str() + datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d_%H_%M_%S")
  # Create a training object
  training = Train(tmp_config, log_dir, save_dir, comments = comments)
  # Train the model
  training.train(Model)  
import tensorflow as tf
from Model import model_base

class GAN_00(model_base.NN_Base):
  def __init__(self, config):
    self.is_training = tf.placeholder(tf.bool, name = 'Is_training')
    super(GAN_00, self).__init__(self.is_training,\
          config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
    self._num_classes = config.NUM_CLASSES
    self._batch_size = config.BATCH_SIZE
    self.config = config
  
  def generator(self, inputs):
    generator_arch = self.config.GENERATOR_NUM_HL
    with tf.name_scope('Generator'):
      with tf.variable_scope('Generator') as scope:
        z = tf.random_normal(shape = (tf.shape(inputs)[0], generator_arch[0]),\
                              mean = 0.0,
                              stddev = 1.0,
                              dtype = tf.float32,
                              name = 'latent_variable'
                            )
        for idx, num_hl in enumerate(generator_arch[1:]):
          with tf.name_scope('Layer_%d'%idx):
            z = self._fully_connected(z, num_hl, use_bias=False)
            z= self._drop_out(z, rate = 0.5)
            z = self._batch_norm(z)
            z = self._softplus(z)
        z = self._dense_WN(z, self.config.FEATURE_NUM)
        z = tf.tanh(z, name = 'Tanh_activation')
    return z
  
  def discriminator(self, inputs, reuse = True):
    ## Todo: reuse the variables in generator
    discriminator_arch = self.config.DISCRIMINATOR_NUM_HL
    with tf.name_scope('Discriminator'):
      with tf.variable_scope('Discriminator', reuse = reuse) as scope:
        for idx, num_hl in enumerate(discriminator_arch):
          with tf.name_scope('Layer_%d'%idx):
            stddev = 1e-5 if idx == 0 else 5e-5
            inputs = tf.cond(self.is_training, lambda:self._add_noise(inputs, stddev = stddev), lambda:inputs)
            inputs = self._dense_WN(inputs, num_hl)
            inputs = self._drop_out(inputs, rate = 0.5)
            inputs = self._leakyrelu(inputs)
        
        with tf.name_scope('Logits_bf_softmax'):
          logits = self._fully_connected(inputs, self.config.NUM_CLASSES)
          tf.add_to_collection("output_logits", logits)
    ## the second return var 'inputs' can be treat as feature in fm loss
    return logits, inputs 
  
  def forward_pass(self, real_lab_input, real_unl_input):
    ## Generator
    fake_input = self.generator(real_lab_input)
    
    ## Discriminator
    d_real_lab_logits, d_real_lab_feat = self.discriminator(real_lab_input, reuse = False)
    d_real_unl_logits, d_real_unl_feat = self.discriminator(real_unl_input, reuse = True)
    d_fake_logits, d_fake_feat = self.discriminator(fake_input, reuse = True)
    
    d_real_unl_logits, d_real_unl_feat = self.discriminator(real_lab_input, reuse = True)
    d_fake_logits, d_fake_feat = self.discriminator(real_lab_input, reuse = True)
    
    ## Todo: add summary op
    
    return d_real_lab_logits, d_real_unl_logits, d_fake_logits,\
           d_real_unl_feat, d_real_lab_feat, d_fake_feat
  
  def forward_pass_for_test(self, val_input):
    logits, _ = self.discriminator(val_input, reuse = False)
    return logits
  

if __name__ == "__main__":
  import sys
  sys.path.append('/home/cdsw')
  import numpy as np
  
  from Config.config import Config
  
  class TempConfig(Config):
    IS_TRAINING = True  
  tmp_config = TempConfig()
  
  tf.reset_default_graph()
  model = GAN_00(tmp_config)
  real_lab_input = tf.random_normal(shape = (tmp_config.BATCH_SIZE, \
                                        tmp_config.GENERATOR_NUM_HL[-1]))
  
  real_unl_input = tf.random_normal(shape = (tmp_config.BATCH_SIZE, \
                                        tmp_config.GENERATOR_NUM_HL[-1]))
  
  real_lab_logits, real_unl_logits, fake_logits, real_unl_feat, fake_feat\
      = model.forward_pass(real_lab_input, real_unl_input)
  
  with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    real_lab_logits_out, real_unl_logits_out, \
    fake_logits_out, real_unl_feat_out, \
    fake_feat_out = sess.run([real_lab_logits, \
                              real_unl_logits, \
                              fake_logits, \
                              real_unl_feat, \
                              fake_feat], \
                              feed_dict = \
                             {model.is_training: tmp_config.IS_TRAINING})
    
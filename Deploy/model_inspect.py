'''
This is a python file that used to inspect your trained ckpt file.
'''
import os, sys
sys.path.append('/home/cdsw')
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
    
class Model_Inspect(object):
  def __init__(self, model_path):
    self.model_path = model_path 
  
  def import_and_restore(self, clear_devices = True):
    saver = tf.train.import_meta_graph(self.model_path+ '.ckpt.meta', \
                                       clear_devices = clear_devices)
    self.graph = tf.get_default_graph()
    
    with tf.Session() as sess:
      saver.restore(sess, self.model_path + '.ckpt')
  
  def check_op(self):  
    # Check all operations (nodes) in the graph:
    print("## All operations: ")
    for op in self.graph.get_operations():
        print(op.name)
  
  def check_var(self):
    # Check all variables in the graph:
    print("## All variables: ")
    for v in tf.global_variables():
        print(v.name)

  def check_trainable(self):      
    # Check all trainable variables in the graph:
    print("## Trainable variables: ")
    for v in tf.trainable_variables():
        print(v.name)
  
  def check_tensor_and_weights(self):
    # Inspect all tensors and their weight values
    reader = pywrap_tensorflow.NewCheckpointReader(self.model_path + '.ckpt')
    var_to_shape_map = reader.get_variable_to_shape_map()
    
    for key in sorted(var_to_shape_map):
      print("tensor_name: ", key)
      print(reader.get_tensor(key))

  
  def save_txt(self, save_path, save_name):
    # Save the whole graph and weights into a text file:
    input_graph_def = self.graph.as_graph_def()
    tf.train.write_graph(input_graph_def, \
                         logdir = save_path, name = save_name, as_text=True)

if __name__=="__main__":
  model_path = 'Deploy/FinalModel/model_final'
  inspect = Model_Inspect(model_path)
  inspect.import_and_restore()
  """
  You may use the following code to inspect your ckpt file
  
  inspect.check_op()
  inspect.check_var()
  inspect.check_trainable()
  save_path = 'Deploy/Log'
  save_name = 'Weights.pbtxt'
  inspect.save_txt(save_path, save_name)
  inspect.check_tensor_and_weights()
  """
  
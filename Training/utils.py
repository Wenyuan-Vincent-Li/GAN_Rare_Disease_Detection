"""
This is a utils python file that can be used by Training session.
"""
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.client import device_lib

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

def variable_name_string():
    name_string = ''
    for v in tf.global_variables():
        name_string += v.name + '\n'
    return name_string

def get_trainable_weight_num(var_list = tf.trainable_variables()):
  total_parameters = 0
  for variable in var_list:
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
      variable_parameters *= dim.value
    total_parameters += variable_parameters
  return total_parameters

def variable_name_string_specified(variables):
  name_string = ''
  for v in variables:
    name_string += v.name + '\n'
  return name_string

def grads_dict(gradients, histogram_dict):    
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient
        histogram_dict[variable.name + "/gradients"] = grad_values
        histogram_dict[variable.name + "/gradients_norm"] =\
                       clip_ops.global_norm([grad_values])
    return histogram_dict

def fn_inspect_checkpoint(ckpt_filepath, **kwargs):
    name = kwargs.get('tensor_name', '')
    if name == '':
        all_tensors = True
    else:
        all_tensors = False    
    chkp.print_tensors_in_checkpoint_file(ckpt_filepath, name, all_tensors)

def save_dict_as_txt(Dict, dir_name):
  if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
  for key in Dict.keys():
    np.savetxt(dir_name + key + '.txt', np.asarray(Dict[key]))

def convert_list_2_nparray(varlist):
    var_np = np.empty((0,))
    for i in range(len(varlist)):
      var_np = np.concatenate((var_np, varlist[i]))
    return var_np
  
def add_grads(grads, histogram_dict): 
  for index, grad in enumerate(grads):
     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
      
      
def initialize_uninitialized_vars(sess):
  from itertools import compress
  global_vars = tf.global_variables()
  local_vars = tf.local_variables()
  init_var = global_vars + local_vars
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in init_var])
  not_initialized_vars = list(compress(global_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))
    
  local_vars = tf.local_variables()
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in local_vars])
  not_initialized_vars = list(compress(local_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))
    
###########################
## Test code
###########################
def _main_variable_name_string():
    print(variable_name_string())

def _main_inspect_checkpoint():
    from Training.Saver import Saver
    save_dir = os.path.join(os.getcwd(), "weight")
    saver = Saver(save_dir)
    _, filename, _ = saver._findfilename()
    fn_inspect_checkpoint(filename)
    fn_inspect_checkpoint(filename, tensor_name = 'conv2d/kernel')

if __name__ == "__main__":
  name = get_available_gpus()
  print(name)
"""
This is a utils python file that can be used by Training session.
"""
import numpy as np
import os
import tensorflow as tf

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
    
def initialize_uninitialized_vars(sess):
  from itertools import compress
  global_vars = tf.global_variables()
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
  not_initialized_vars = list(compress(global_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))
    
  local_vars = tf.local_variables()
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in local_vars])
  not_initialized_vars = list(compress(local_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))

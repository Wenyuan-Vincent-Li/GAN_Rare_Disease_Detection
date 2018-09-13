'''
This is a evaluation base function upon which the Evaler.py was built.
'''
import tensorflow as tf

class Evaler_base(object):
  def __init__(self):
    pass
    
  def _accuracy_metric(self, labels, predictions):
    return tf.metrics.accuracy(labels, predictions)
  
  def _auc_metric(self, labels, predictions, curve = 'ROC'):
    return tf.metrics.auc(labels, predictions, curve = curve)
  
  def _streaming_auc_metric(self, labels, predictions, curve = 'ROC'):
    return tf.contrib.metrics.streaming_auc(predictions, labels, curve = curve)
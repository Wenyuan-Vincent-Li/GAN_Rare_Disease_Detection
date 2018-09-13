'''
This is a training base function upon which the Train.py was built.
'''
import tensorflow as tf
import sys
sys.path.append('/home/cdsw')
import Training.utils as utils

class Train_base(object):
    def __init__(self, learning_rate, momentum = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def _input_fn(self):
        raise NotImplementedError(
                'metirc() is implemented in Model sub classes')
    
    def _build_train_graph(self):
        raise NotImplementedError(
                'loss() is implemented in Model sub classes')
    
    def _loss(self, target, network_output):
        raise NotImplementedError(
                'loss() is implemented in Model sub classes')
    
    def _loss_weight_l2(self, var_list, eta = 0.001):
      num_weights = utils.get_trainable_weight_num(var_list)
      loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list]) * eta / num_weights
      return loss
    
    def _huber_loss(self, labels, predictions, delta=1.0, name = "huber_loss"):
      with tf.name_scope(name):
        residual = tf.abs(predictions - labels)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
      return tf.where(condition, small_res, large_res)
    
    def _entropy(self, logits):
      with tf.name_scope('Entropy'):
        probs = tf.nn.softmax(logits)
        ent = tf.reduce_mean(- tf.reduce_sum(probs * logits, axis = 1, keep_dims = True) \
              + tf.reduce_logsumexp(logits, axis = 1, keep_dims = True))
      return ent
    
    def _metric(self, labels, network_output):
        raise NotImplementedError(
                'metirc() is implemented in Model sub classes')
    
    def _train_op(self, optimizer, loss):
        train_op = optimizer.minimize(loss, 
                                      global_step = tf.train.get_global_step())
        return train_op
    
    def _train_op_w_grads(self, optimizer, loss, var_list = None):
        grads = optimizer.compute_gradients(loss, var_list = var_list)
        train_op = optimizer.apply_gradients(grads)
        return train_op, grads
        
    def _cross_entropy_loss(self, labels, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, \
                                                          logits = logits)
        loss = tf.reduce_mean(loss)
        return loss
    
    def _SGD_w_Momentum_optimizer(self):
        optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate,
                                               momentum = self.momentum)
        return optimizer
    
    def _Adam_Optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        return optimizer
    
    def _accuracy_metric(self, labels, predictions):
        return tf.metrics.accuracy(labels, predictions)
    
    def _auc_metric(self, labels, predictions, curve = 'ROC'):
        return tf.metrics.auc(labels, predictions, curve = curve)
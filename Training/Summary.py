'''
This is a python file that creats a Summary object that can be visulized by tensorboard.
'''
import tensorflow as tf
import os
from time import strftime
from datetime import datetime
from pytz import timezone
import sys
sys.path.append(os.path.dirname(os.getcwd()))

class Summary(object):
    def __init__(self, log_dir, config, **kwargs):
        self.config = config
        if kwargs is not None:
            if 'log_comments' in kwargs:
                self.comments = kwargs.get('log_comments')
            if 'log_type' in kwargs:
                log_dir = os.path.join(log_dir, kwargs.get('log_type'))
        
        log_dir = os.path.join(log_dir, 'Run_' + \
                               datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d_%H_%M_%S"))
        
        if not os.path.dirname(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir        
        self.summary_writer = tf.summary.FileWriter(log_dir)
        self._write_comments()
                
    
    def add_summary(self, summary_dict):
        tf.logging.debug("come into site 1")
        with tf.name_scope('summary'):
            if 'scalar' in summary_dict:
                self._scalar_summary(summary_dict['scalar'])
            if 'image' in summary_dict:
                self._image_summary(summary_dict['image'])
            if 'histogram' in summary_dict:
                self._histogram_summary(summary_dict['histogram'])
                
            merged_summary = tf.summary.merge_all()
        return merged_summary
        
    def _graph_summary(self, graph):
        self.summary_writer.add_graph(graph)
        return
    
    def _scalar_summary(self, scalar_dict):
        for name, value in scalar_dict.items():
            tf.summary.scalar(name, value)
        return
    
    def _histogram_summary(self, histogram_dict):
        for name, value in histogram_dict.items():
            tf.summary.histogram(name.replace(':', '_'), value)
        return
    
    def _image_summary(self, image_dict, max_outputs = 2):
        for name, image in image_dict.items():    
            tf.summary.image(name, image, max_outputs)
        return
    
    def _write_comments(self):
        with open(os.path.join(self.log_dir, 'Comments.txt'), 'w') as txt_file:
            txt_file.write(self.comments)
    
    def _image_parser(self):
        raise NotImplementedError(
                '_image_parser is implemented in Model sub classes')
        return
    

if __name__ == "__main__":
    from config import Config
    comments = 'this is a test txtfile'
    log_dir = os.path.join(os.getcwd(),"temp")
    
    class TestConfig(Config):
        IS_TRAINING = True
        DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "Dataset")
    
    config = TestConfig    
    summary_train = Summary(log_dir, config, log_type = 'train', \
                                             log_comments = comments)
    summary_train = Summary(log_dir, config, log_type = 'val', \
                            log_comments = comments)
    summary = Summary(log_dir, config, log_comments = 'single summary')
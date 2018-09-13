'''
This is a python file that used for deploy a trained model for API serving.
'''
import os, sys
sys.path.append('/home/cdsw')
import pandas as pd
import tensorflow as tf
from Deploy.deploy_base import Deploy_base

class Deploy(Deploy_base):
  def __init__(self, model_path, save_path, **kwargs):
    super(Deploy, self).__init__(model_path, save_path)
  
  def extend_meta_graph(self, clear_devices = True):
    """
    Extend and modify the meta graph
    """
    # import the meta graph
    self.tf_graph = self._import_meta_graph(clear_devices)

    # add nodes to the metagraph
    """
    You may use the following sample code to debug and extend the graph
    
    logits = tf.get_collection('logits')[0]
    prediction = tf.argmax(output_logits, axis = -1, name = 'prediction')
    probs = tf.nn.softmax(output_logits, name = 'probs') 
    collections_keys = self.tf_graph.get_all_collection_keys()
    tf.logging.debug(collections_keys)
    tf.logging.debug(self.tf_graph.get_tensor_by_name('probs:0'))
    
    """
    return
  
  def freeze_model(self):
    # Check the name in your default graph
    tf.logging.debug([n.name for n in \
                            tf.get_default_graph().as_graph_def().node])
    
    # Specify the name of your output variable, and seperate them by ','
    # These output variables will be used as the output of your frozen model
    output_node_names = "Results/ArgMax,Results/strided_slice"
    
    # Freeze the model
    self._freeze_model(self.save_path, output_node_names)
  
  def use_frozen_model(self, config):
    # Load the frozen model
    self.tf_graph = self._load_frozen_model()
    # Check each opretion name
    tf.logging.debug([op.name for op in self.tf_graph.get_operations()])
    # Get the input data
    self._input_fn(config)
    # Access the input and output nodes 
    input_feature = self.tf_graph.get_tensor_by_name('prefix/input_feature:0')
    input_istraining = self.tf_graph.get_tensor_by_name('prefix/Is_training:0')
    preds = self.tf_graph.get_tensor_by_name('prefix/Results/ArgMax:0')
    probs = self.tf_graph.get_tensor_by_name('prefix/Results/strided_slice:0')
    # Run the prediction
    with tf.Session(graph=self.tf_graph) as sess:
      # Note: we don't need to initialize/restore anything
      # There is no Variables in this graph, only hardcoded constants 
      preds_o, probs_o = sess.run([preds, probs], feed_dict={
            input_feature: self.input_array,
            input_istraining: False
      })
    return preds_o, probs_o
  
  def _input_fn(self, config):
    # Read in the input data
    data_api = pd.read_csv(config.DATA_FILE)
    data_api = data_api.drop('OUTCOME', axis = 1)
    self.input_array = data_api.values

def _main_freeze_model():
  """
  Main function for freezing the trained model
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs
  
  # Specify the path saved the model
  model_path = "Deploy/FinalModel/model"
  # Specify the path you want save the frozen model
  save_path = "Deploy/ModelWrapper"                              
  
  # Freeze the model
  deploy = Deploy(model_path, save_path)
  deploy.extend_meta_graph()
  deploy.freeze_model()

def _serve_freeze_model():
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable all debugging logs
  tf.logging.set_verbosity(tf.logging.INFO)
  
  from Config.config import Config
  class TempConfig(Config):
    DATA_FILE = "Data/Data_File/API_SERVER_SAMPLE.csv" # File path of your input data
  
  # Create a global configuration object
  tmp_config = TempConfig()
  
  model_path = "" # Don't need to specify this in this scenario.
  # Specify the path you saved rozen model
  save_path = "Deploy/ModelWrapper"                              
  
  # Use the frozen model for prediction
  deploy = Deploy(model_path, save_path)
  preds, probs = deploy.use_frozen_model(tmp_config)

if __name__ == '__main__':
  _main_freeze_model()
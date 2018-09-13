'''
This is a deploy base function upon which the Deploy.py was built.
'''
import sys
import os
sys.path.append('/home/cdsw')
import tensorflow as tf


class Deploy_base(object):
  def __init__(self, model_path, save_path):
    self.model_path = model_path
    self.save_path = save_path
    tf.reset_default_graph()
    
  
  def _import_meta_graph(self, clear_devices = True):
    # We import the meta graph in the current default Graph
    self.saver = tf.train.import_meta_graph(self.model_path + '.ckpt.meta', \
                                       clear_devices = clear_devices)
    return tf.get_default_graph()
  
  def extend_meta_graph(self):
    raise NotImplementedError('_input_fn() is implemented in Model sub classes')
  
  def _freeze_model(self, save_path, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        save_path: the root folder to save the frozen model
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not output_node_names:
      print("You need to supply the name of a node to --output_node_names.")
      return -1
    output_graph = os.path.join(save_path, \
                   os.path.basename(self.model_path) + "_frozen.pb")
    with tf.Session() as sess: 
      # We restore the weights
      self.saver.restore(sess, self.model_path + '.ckpt')
      # We use a built-in TF helper to export variables to constants
      output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), 
        # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") 
        # The output node names are used to select the usefull nodes
      ) 
      # Finally we serialize and dump the output graph to the filesystem
      with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
      print("Genearting the frozen graph. %d ops in the final graph."\
            % len(output_graph_def.node))
    return
  
  def _load_frozen_model(self):
    """
    Load the frozen pb model.
    """
    frozen_graph_filename = os.path.join(self.save_path, "model_frozen.pb")
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph
  
  def _input_fn(self):
    """
    Create the input pipeline for the reference model
    """
    raise NotImplementedError('_input_fn() is implemented in Model sub classes')
    
if __name__ == '__main__':      
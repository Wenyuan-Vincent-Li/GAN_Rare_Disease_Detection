"""
This is a python file that used flask and serve the trained model as a web API.
"""
import sys
import os
sys.path.append('/home/cdsw')

import flask
from flask import render_template, send_from_directory, request

import tensorflow as tf 
import numpy as np
import pandas as pd

th_1 = 0.35 # Below which, the risk level is low
th_2 = 0.7 # More than which, the risk level is high

# Obtain the flask app object
app = flask.Flask(__name__)


def color_probs(val):
  '''
  Function that colors the content based on different condition
  '''
  if val < th_1:
    color = 'green'
  elif val > th_2:
    color = 'red'
  else:
    color = 'darkorange'
  return 'color: %s' % color

def background_color(val):
  '''
  Function that assign different background color based on the content
  '''
  if val == 'Low Risk':
    color = 'lightgreen'
  elif val == 'High Risk':
    color = 'red'
  else:
    color = 'darkorange'
  return 'background-color: %s' % color

def highlight_high_risk(s):
  '''
  Function that hightlight the High Risk
  '''
  is_highrisk = s == 'High Risk'
  return ['background-color: red' if v else '' for v in is_highrisk]

def post_processing(raw_data, probs):
  '''
  Function that do post processing
  '''
  ## First insert the prediction results and save the pandas as csv file
  raw_data.insert(0, "DISEASE_PROBS", probs, False)
  raw_data.insert(0, "RISK_LEVEL", 0, False)
  raw_data.loc[raw_data['DISEASE_PROBS'] > th_2, "RISK_LEVEL"] = 'High Risk'
  raw_data.loc[raw_data['DISEASE_PROBS'] < th_1, "RISK_LEVEL"] = 'Low Risk'
  raw_data.loc[(raw_data['DISEASE_PROBS'] < th_2) & \
               (raw_data['DISEASE_PROBS'] > th_1), "RISK_LEVEL"] = 'Med Risk'
  raw_data.to_csv(os.path.join('~/Serving/tmp', 'crt.csv'),\
                  index = False)
  
  ## Then render the pandas as html in certain style
  data = raw_data.loc[:9, 'RISK_LEVEL': 'AB_HERN_FREQ']
  data_html = data.style.\
              bar(subset=['DISEASE_PROBS'], color='#d65f5f').\
              applymap(background_color, subset=['RISK_LEVEL']).\
              hide_index().\
              set_properties(**{'text-align': 'center'}).\
              set_table_attributes("border=1").\
              format("{:.2f}", \
                     subset= ['PAT_AGE', 'AB_PEL_PAIN_COUNT', 'AB_PEL_PAIN_FREQ',
                             'AB_PEL_PAIN_EXP_TD', 'AB_HERN_COUNT', 'AB_HERN_FREQ']).\
              format({'DISEASE_PROBS': "{:.2%}"}).render()
  
  return data_html

def load_graph(model_path):
  # We load the protobuf file from the disk and parse it to retrieve the 
  # unserialized graph_def
  with tf.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we import the graph_def into a new Graph and returns it 
  with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="prefix")
  
  return graph

  
def input_processing(file_path):
  # Reading the csv file and do pre-processing
  # Note that in the real application, you need to do the pre-processing in the same way
  # we did when training the GANs model on your own data. Make sure you use the same scaling
  # factor as we used for the training data. The resulted input feature should be 
  # [None, 797]. If you don't know the process of the pre-processing, you can take a look at
  # the Data folder.
  # Failure to do the pre-processing will give you the bad results.
  
  file_path = os.path.join('/home/cdsw', file_path)
  data_api = pd.read_csv(file_path)
  data_api = data_api.drop('OUTCOME', axis = 1)
  input_array = data_api.values
  
  return input_array, data_api

  
@app.route('/') # The homepage
def index():
    return render_template("index.html")


@app.route('/predict/',methods=['POST', 'GET']) # The prediction page
def predict():
  if request.method == 'POST':
    file_path = request.form['file_path']
    if file_path == '': # Check the user input the file_path
      return render_template("errorpage.html",\
                             message = 'Please specify csv file path!')
    try: # Check the user input the right file_path
      input_array, raw_data = input_processing(file_path)
    except Exception as e: 
      return render_template("errorpage.html", \
                            message = 'There is no such file or the file can not be read in!')
    
    # Fetch the graph and do the prediction
    graph =app.graph
    input_feature = graph.get_tensor_by_name('prefix/input_feature:0')
    input_istraining = graph.get_tensor_by_name('prefix/Is_training:0')
    preds = graph.get_tensor_by_name('prefix/Results/ArgMax:0')
    probs = graph.get_tensor_by_name('prefix/Results/strided_slice:0')
    with tf.Session(graph = graph) as sess:
      # Note: we don't need to initialize/restore anything
      # There is no Variables in this graph, only hardcoded constants 
      preds_o, probs_o = sess.run([preds, probs], feed_dict={
          input_feature: input_array,
          input_istraining: False
      })
    # Do the post data processing  
    data = post_processing(raw_data, probs_o)
    return render_template("results.html", data = data)
  
@app.route('/download/',methods=['POST', 'GET']) # Download page
def download():
  return send_from_directory('./tmp', 'crt.csv', as_attachment=True)


# Only load the graph once, when start serve the api
app.graph=load_graph('./model/model_frozen.pb')  

if __name__ == '__main__':
  app.secret_key = 'super secret key'
  app.config['SESSION_TYPE'] = 'filesystem'
  app.run(host=os.environ["CDSW_IP_ADDRESS"],
          port=int(os.environ["CDSW_PUBLIC_PORT"]),
          debug=True)
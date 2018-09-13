import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/cdsw')
th_1 = 0.35
th_2 = 0.7

def color_probs(val):
  if val < th_1:
    color = 'green'
  elif val > th_2:
    color = 'red'
  else:
    color = 'darkorange'
  return 'color: %s' % color

def background_color(val):
  if val == 'Low Risk':
    color = 'lightgreen'
  elif val == 'High Risk':
    color = 'red'
  else:
    color = 'darkorange'
  return 'background-color: %s' % color

def highlight_high_risk(s):
  is_highrisk = s == 'High Risk'
  return ['background-color: red' if v else '' for v in is_highrisk]

def post_processing(raw_data, probs):
  raw_data.insert(0, "DISEASE_PROBS", probs, False)
  raw_data.insert(0, "RISK_LEVEL", 0, False)
  raw_data.loc[raw_data['DISEASE_PROBS'] > th_2, "RISK_LEVEL"] = 'High Risk'
  raw_data.loc[raw_data['DISEASE_PROBS'] < th_1, "RISK_LEVEL"] = 'Low Risk'
  raw_data.loc[(raw_data['DISEASE_PROBS'] < th_2) & \
               (raw_data['DISEASE_PROBS'] > th_1), "RISK_LEVEL"] = 'Med Risk'
  raw_data.to_csv(os.path.join('~/Serving/tmp', 'crt.csv'),\
                  index = False)
  
  data = raw_data.loc[:10, 'RISK_LEVEL': 'AB_PEL_PAIN_FREQ']
  data_style = data.style.\
              bar(subset=['DISEASE_PROBS'], color='#d65f5f').\
              applymap(background_color, subset=['RISK_LEVEL']).\
              hide_index().\
              set_properties(**{'text-align': 'center'}).\
              set_table_attributes("border=1").\
              format("{:.2f}", \
                     subset= ['PAT_AGE', 'AB_PEL_PAIN_COUNT', 'AB_PEL_PAIN_FREQ']).\
              format({'DISEASE_PROBS': "{:.2%}"})
  return data_style

if __name__ == "__main__":
  path = "Data/Data_File/API_SERVER_SAMPLE.csv"
  raw_data = pd.read_csv(path)
  raw_data = raw_data.drop("OUTCOME", axis = 1)
  probs = np.random.uniform(0, 1, raw_data.shape[0])
  data_style = post_processing(raw_data, probs)
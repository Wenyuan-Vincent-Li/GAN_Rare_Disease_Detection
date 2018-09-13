'''
This is a python file that used for creating bechmark for GAN.
'''
## Import Module
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve,\
                            matthews_corrcoef, f1_score, log_loss, roc_auc_score

## Read in data
raw_train = pd.read_csv('Data/Data_File/Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv')
raw_test = pd.read_csv('Data/Data_File/Test_AFT_FS_All_MinMax_SHUFFLE.csv')

## Define X and y
target = 'OUTCOME'

fea_use = raw_test.columns[1:] # Defien the used features
fea_test = raw_test.columns

data_train = raw_train[raw_train['Lab_Used_In_Train'] == 1]
temp_0 = raw_train[raw_train['Lab_Used_In_Train'] == 0][fea_test]
data_test = pd.concat([temp_0, raw_test], axis = 0)  # Combine unlabeld data with test data

surrogate = 0 # fill the missing value to be 0
X_train = data_train.fillna(surrogate)[fea_use].values
y_train = data_train.fillna(surrogate)[target].values

X_test = data_test.fillna(surrogate)[fea_use].values
y_test = data_test.fillna(surrogate)[target].values

## Bechmark
# 1. define classifier
c1 = xgb.XGBClassifier(n_estimators = 300) # XGB
c2 = RandomForestClassifier(n_estimators = 300) # RF
c3 = LogisticRegression(max_iter=100) # LR
c4 = MLPClassifier(solver = 'sgd', alpha = 1e-5,
                  hidden_layer_sizes = (1000, 500, 250, 250, 250),
                  random_state = 1)

# 2. fit classifier to data
print('Start training XGB')
t1 = time.time()
c1.fit(X_train, y_train)
print('XGB model uses %f secs' % (time.time() - t1) ) # 128s

print('Start training RF')
t2 = time.time()
c2.fit(X_train, y_train)
print('RF model uses %f secs' % (time.time() - t2) ) # 68s

print('Start training LR')
t3 = time.time()
c3.fit(X_train, y_train)
print('LR model uses %f secs' % (time.time() - t3) ) # 22s

print('Start training NN')
t4 = time.time()
c4.fit(X_train, y_train)
print('NN model uses %f secs' % (time.time() - t3) ) # 342s

# 3. make prediction
p1 = c1.predict_proba(X_test)[:,1]
p2 = c2.predict_proba(X_test)[:,1]
p3 = c3.predict_proba(X_test)[:,1]
p4 = c4.predict_proba(X_test)[:,1]
PS = np.array([p1, p2, p3, p4])# prediction probability scores

## plot curves
LL = ['XGB', 'RF', 'LR', 'NN']
timestr = time.strftime("%Y%m%d-%H%M%S")

# 1. PR curve
def show_PR():
    plt.figure(1)
    LL_c = LL
    for i, item in enumerate(LL_c):
        y_score = PS[i]
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score, average="micro")
        plt.plot(recall, precision, label= item + ' PR_AUC:%f' % pr_auc)
        if True: # save the data
          pr = pd.DataFrame({'Precision': precision, 'Recall': recall})
          pr.to_csv('Plots/Results_WY/PR_CURVE_'+ item + '.csv')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    
    plt.savefig('Temp/compare PR curve_'+ timestr + '.png')
    plt.show()
show_PR()

# 2. ROC AUC analysis
def show_roc():
  plt.figure(2)
  plt.plot([0, 1], [0, 1], 'k--')
  for i, item in enumerate(LL):
      y_score = PS[i]
      ras = roc_auc_score(y_test, y_score)
      fpr, tpr, _ = roc_curve(y_test, y_score)
      plt.plot(fpr, tpr, label= item + ' ROC_AUC:%f' % ras)


  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('ROC Curves' )
  plt.legend(loc="lower right")
  plt.savefig('Temp/compare ROC curve_'+ timestr + '.png')
  plt.show()
show_roc()


## other analysis
# 1. feature importance analysis
fea_weight = pd.DataFrame({'FEATURE_NAME': features, 'IMPORTANCE': c1.feature_importances_})
fea_weight.to_csv('Temp/ratio_3_feature importance_'+ timestr + '.csv')
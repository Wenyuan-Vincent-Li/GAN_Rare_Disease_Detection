import xgboost as xgb
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef, f1_score
import time
from sklearn.cross_validation import train_test_split

#res_old = pd.read_csv('For_Yunlong/raw_data/model_data_all.csv')
#res = pd.read_csv('For_Yunlong/raw_data/model_data_all_v2.csv')


raw_train = pd.read_csv('Data/Data_File/Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv')
raw_test = pd.read_csv('Data/Data_File/Test_AFT_FS_All_MinMax_SHUFFLE.csv')


target = 'OUTCOME'

fea_use = raw_test.columns[1:]
fea_test = raw_test.columns

data_train = raw_train[raw_train['Lab_Used_In_Train'] == 1]
temp_0 = raw_train[raw_train['Lab_Used_In_Train'] == 0][fea_test]
data_test = pd.concat([temp_0, raw_test], axis = 0)  


################################
# define X and y
################################

features = fea_use


surrogate = 0
X_train = data_train.fillna(surrogate)[features].values
y_train = data_train.fillna(surrogate)[target].values


X_test = data_test.fillna(surrogate)[features].values
y_test = data_test.fillna(surrogate)[target].values




################################
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef, f1_score
import time
from sklearn.ensemble import ExtraTreesClassifier

# 1. define classifier

c1 = xgb.XGBClassifier(n_estimators = 300)
c2 = RandomForestClassifier(n_estimators = 300)
c3 = LogisticRegression(max_iter=100)


# 2. fit classifier to data

print('start training')
t1 = time.time()
c1.fit(X_train, y_train)
print('XGB model uses %f secs' % (time.time() - t1) )

print('start training')
t2 = time.time()
c2.fit(X_train, y_train)
print('RF model uses %f secs' % (time.time() - t2) )

print('start training')
t3 = time.time()
c3.fit(X_train, y_train)
print('LR model uses %f secs' % (time.time() - t3) )



# 3. make prediction



p1 = c1.predict_proba(X_test)[:,1]
p2 = c2.predict_proba(X_test)[:,1]
p3 = c3.predict_proba(X_test)[:,1]

PS = np.array([p1, p2, p3])# prediction probability scores


############################
## plot curves
############################

LL = ['XGB', 'RF', 'LR',]
import matplotlib.pyplot as plt
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

from sklearn.metrics import roc_auc_score, roc_curve

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



############################
## csv files
############################
# 1. feature importance analysis
fea_weight = pd.DataFrame({'FEATURE_NAME': features, 'IMPORTANCE': c1.feature_importances_})
fea_weight.to_csv('Temp/ratio_3_feature importance_'+ timestr + '.csv')















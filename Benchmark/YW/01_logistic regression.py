import pandas as pd
import numpy as np

#data_train = pd.read_csv('Data/Data_File/Train_AFT_FS_All.csv')
#data_test = pd.read_csv('Data/Data_File/Test_AFT_FS_All.csv')

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from __future__ import print_function
from datetime import datetime
from time import time
import numpy as np


# initialize spark session
conf = SparkConf() \
    .set("spark.dynamicAllocation.initialExecutors", "8") \
    .set("spark.dynamicAllocation.maxExecutors", "64") \
    .set("spark.executor.memoryOverhead", "3g") \
    .set("spark.kryoserializer.buffer.max", "1g") \
    .set("spark.app.name", "KYYOTA") \
    .set("spark.executor.memory", "20g") \
    .set("spark.driver.memory", "16g") \
    .set("spark.driver.cores", "32") \
    .set("spark.yarn.queue", "ace")

    
spark = SparkSession.builder.appName("KYYOTA").config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel("FATAL")    

HDFS_PATH = '/user/yunlong.wang/Abbvie_EPI/temp'
data_train = spark.read.csv('{}/Train_AFT_FS_All.csv'.format(HDFS_PATH), inferSchema=True, header=True)
data_test = spark.read.csv('{}/Test_AFT_FS_All.csv'.format(HDFS_PATH), inferSchema=True, header=True)


!hadoop fs -copyFromLocal Data/Data_File/xxx /user/wenyuan.li/xxx/xxx

!hadoop fs -copyToLocal /user/wenyuan.li/xxx/xxx Data/Data_File/xxx



print (data_train.columns)


data_train.groupBy('OUTCOME').count().show()

#|OUTCOME|  count|
#+-------+-------+
#|      1|   5923|
#|      0|1184600|
#+-------+-------+



from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression

from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef, f1_score
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, IndexToString 
import time
import matplotlib.pyplot as plt
import datetime




# II. define feature columns

fea_use = data_train.columns[2:]


# III. assemble feature columns to one column

assembler = VectorAssembler(inputCols = fea_use, outputCol="features")
train_2 = assembler.transform(data_train)
test_2 = assembler.transform(data_test)

test_2.printSchema() 
# from this you will see the last column is |-- features: vector (nullable = true) 


# IV. train model

# ----------------------------------------------------------------------
## Model training and prediction
start_time = time.time()
print('model training start at: ', datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
lr = LogisticRegression(labelCol="OUTCOME", featuresCol="features", maxIter=100)

### Fit the model on training data.
trained_model_lr = lr.fit(train_2)

print('model training completed at: ', datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
m, s = divmod(time.time() - start_time, 60)
h, m = divmod(m, 60)
print('model training run time: %d:%02d:%02d' % (h, m, s))

lr.save("Benchmark/trained_model/")


# V. Make predictions on test data


pred_test = trained_model_lr.transform(test_2)

evaluator = BinaryClassificationEvaluator(labelCol="OUTCOME", rawPredictionCol="rawPrediction")
auroc = evaluator.evaluate(pred_test, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(pred_test, {evaluator.metricName: "areaUnderPR"})
print("The ROC_AUC is %.4f and the PR_AUC is %.4f" % (auroc, aupr))

# The ROC_AUC is 0.8365 and the PR_AUC is 0.3634


# VI. Draw curve with test data



predictions_collect = pred_test.select("OUTCOME", "probability",'prediction').collect()
predictions_list = [(float(i[0]), float(i[1][1]), float(i[2])) for i in predictions_collect]

predictions_list_np = np.array(predictions_list)
res = pd.DataFrame(predictions_list_np)
res.columns = ['y_test','y_score','y_pred']



y_test = res['y_test'].values
PS = [ res['y_score'].values ] 

# 1. PR curve
LL_c = ['ACOE']
def plot_fig():
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
  plt.legend(loc="lower left")

  plt.show()

plot_fig()
plt.savefig('Plots/PR_Logistic_regression.png')





# 2. ROC AUC analysis

from sklearn.metrics import roc_auc_score, roc_curve

def plot_fig_ROC():

  plt.plot([0, 1], [0, 1], 'k--')
  for i, item in enumerate(LL_c):
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
  plt.show()

plot_fig_ROC() 











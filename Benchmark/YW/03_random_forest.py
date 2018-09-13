from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from __future__ import print_function
from datetime import datetime
from time import time
import numpy as np
import pandas as pd


# initialize spark session
conf = SparkConf() \
    .set("spark.dynamicAllocation.initialExecutors", "8") \
    .set("spark.dynamicAllocation.maxExecutors", "64") \
    .set("spark.yarn.executor.memoryOverhead", "3g") \
    .set("spark.kryoserializer.buffer.max", "1g") \
    .set("spark.app.name", "KYYOTA") \
    .set("spark.executor.memory", "20g") \
    .set("spark.driver.memory", "16g") \
    .set("spark.driver.cores", "32") \
    .set("spark.yarn.queue", "ace")

    
spark = SparkSession.builder.appName("KYYOTA").config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")    

HDFS_PATH = '/user/fzhang1/rare_disease'
raw_train = spark.read.csv('{}/Train_AFT_FS_All_MinMax_FLAG_SHUFFLE.csv'.format(HDFS_PATH), inferSchema=True, header=True)
raw_test = spark.read.csv('{}/Test_AFT_FS_All_MinMax_SHUFFLE.csv'.format(HDFS_PATH), inferSchema=True, header=True)



#print (data_train.columns)

fea_use = raw_test.columns[1:]
fea_test = raw_test.columns




data_train = raw_train.filter(col('Lab_Used_In_Train') == 1)

temp_0 = raw_train.filter(col('Lab_Used_In_Train') == 0).select(fea_test)
data_test = raw_test.union(temp_0)






from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef, f1_score
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, IndexToString 
import time
import matplotlib.pyplot as plt
import datetime




# II. define feature columns

fea_use = fea_use
print (fea_use.__len__())

# III. assemble feature columns to one column

assembler = VectorAssembler(inputCols = fea_use, outputCol="features")
train_2 = assembler.transform(data_train)
test_2 = assembler.transform(data_test)

test_2.printSchema() 
# from this you will see the last column is |-- features: vector (nullable = true) 


# IV. train model

label = "OUTCOME"



# ----------------------------------------------------------------------
## Model training and prediction
start_time = time.time()
print('model training start at: ', datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
clf_rf = RandomForestClassifier(labelCol= label,
                                featuresCol="features",
                                numTrees=200,
                                maxBins=1024,
                                maxDepth=15)

### Fit the model on training data.
trained_model_rf = clf_rf.fit(train_2)

print('model training completed at: ', datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
m, s = divmod(time.time() - start_time, 60)
h, m = divmod(m, 60)
print('model training run time: %d:%02d:%02d' % (h, m, s))

# model training run time: 0:07:50

#trained_model_rf.save("Benchmark/trained_model/")



# V. Make predictions on test data


 
pred_test_rf = trained_model_rf.transform(test_2)

evaluator = BinaryClassificationEvaluator(labelCol= label, rawPredictionCol="rawPrediction")
auroc = evaluator.evaluate(pred_test_rf, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(pred_test_rf, {evaluator.metricName: "areaUnderPR"})
print("The ROC_AUC is %.4f and the PR_AUC is %.4f" % (auroc, aupr))

# The ROC_AUC is 0.7840 and the PR_AUC is 0.3714





# VI. Draw curve with test data



predictions_collect = pred_test_rf.select(label, "probability",'prediction').collect()
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
plt.savefig('Plots/PR_random_forest_200.png')





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

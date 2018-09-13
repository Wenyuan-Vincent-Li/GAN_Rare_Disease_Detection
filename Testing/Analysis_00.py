import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, \
      average_precision_score, matthews_corrcoef, f1_score,\
      roc_auc_score, roc_curve, accuracy_score, auc

def plot_PR_fig(y_test, y_score):
  precision, recall, thresholds = precision_recall_curve(y_test, y_score)
  pr_auc_metrics = auc(recall, precision)
  pr_auc_score = average_precision_score(y_test, y_score, average="micro")
  plt.plot(recall, precision, label= 'PR_AUC:%f' % pr_auc_metrics)

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall Curve')
  plt.legend(loc="lower left")

  plt.show()
  return pr_auc_score
  
def plot_ROC_fig(y_test, y_score):
  plt.plot([0, 1], [0, 1], 'k--')
  fpr, tpr, _ = roc_curve(y_test, y_score)
  roc_auc_metrics = auc(fpr, tpr)
  plt.plot(fpr, tpr, label= 'ROC_AUC:%f' % roc_auc_metrics)


  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('ROC Curves' )
  plt.legend(loc="lower right")
  plt.show()

def IOU(val_lab, preds):
    TP = []
    for idx in range(len(val_lab)):
      if (val_lab[idx] and preds[idx]):
        TP.append(idx)
    return len(TP) / (sum(preds) + sum(val_lab) - len(TP)), \
           len(TP), len(TP) / sum(preds)

def Analysis(Val_lab, Preds, Probs):
  # 1. Compute accuracy
  acc_score = accuracy_score(Val_lab, Preds)

  # 2. PR curve
  y_score = Probs
  pr_auc_score = plot_PR_fig(Val_lab, y_score)

  # 3. ROC curve
  plot_ROC_fig(Val_lab, y_score)
  
  # 4. IOU
  iou, TP, ratio = IOU(Val_lab, Preds)
  
  # 5. Print Results
  print("Accuracy: {:.2f}, PR_AUC: {:.2f}".format(acc_score, pr_auc_score))
  print("IOU: {:.2f}, TP: {:}, ration{:.2f}".format(iou, TP, ratio))
  return


if __name__ == "__main__":
  file_path = 'Testing/Results_Classifier/epoch=20/'
  Val_lab = np.loadtxt(file_path + 'Val_lab' + '.txt')
  Preds = np.loadtxt(file_path + 'Preds' + '.txt')
  Probs = np.loadtxt(file_path + 'Probs' + '.txt')
  Analysis(Val_lab, Preds, Probs)
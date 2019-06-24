# Copyright (c) Microsoft. All rights reserved.
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc, roc_auc_score
from seqeval.metrics import classification_report


def compute_acc(predicts, labels):
    return 100.0 * accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return 100.0 * f1_score(labels, predicts)

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof

def compute_auc(predicts, labels):
    labels = np.asarray(labels)
    predicts = np.asarray(predicts)
    predicts = np.reshape(predicts, (labels.shape[0], -1))[:, 1]
    fp_rate, tp_rate, thresholds = roc_curve(labels, predicts)
    score = auc(fp_rate, tp_rate)
    return 100.0 * score

def compute_ener(predicts, labels):
    y_true, y_pred = [], []
    from .label_map import NER_LabelMapper
    def trim(predict, label):
        temp_1 =  []
        temp_2 = []
        for j, m in enumerate(predict):
            if j == 0:
                continue
            if NER_LabelMapper[label[j]] != 'X':
                temp_1.append(NER_LabelMapper[label[j]])
                temp_2.append(NER_LabelMapper[m])
            #else:
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)
        print(y_true)
        print(y_pred)
        #break
    for predict, label in zip(predicts, labels):
        trim(predict, label)
    report = classification_report(y_true, y_pred,digits=4)
    return report
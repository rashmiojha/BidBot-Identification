from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt;  plt.rcdefaults()
from DataProcess import Data

from sklearn.cross_validation import StratifiedKFold
from scipy import interp
import numpy as np

d = Data('data')
bid_data = d.bidData
train_data = d.trainData

def allGraphs():
    # train_bids = pd.merge(bid_data, train_data,how='left',on='bidder_id')
    # statData = train_bids.groupby('outcome')['bid_id'].count()
    # '0 =  Human 1 = bot'
    # per_human = statData[0] *100/(statData[0]+statData[1])
    # per_bot = statData[1] *100/(statData[0]+statData[1])
    # objects = ('% Human bids', '% Bot bids')
    # y_pos = np.arange(len(objects))
    # percent = [per_human, per_bot]
    # plt.bar(y_pos, percent, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Percentage of Bids')
    # plt.show()

    bid_data['time'] = bid_data['time'] /100000000000
    plt.scatter(bid_data['time'], bid_data['bid_id'], marker='o', color='b')
    plt.show()

# allGraphs()

def roc_auc(train_features, classifier):
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X = np.array(X_train)
    y = np.array(Y_train)
    cv = StratifiedKFold(y, n_folds=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (tran, tet) in enumerate(cv):
        probas_ = classifier.fit(X[tran], y[tran]).predict_proba(X[tet])
        fpr, tpr, thresholds = roc_curve(y[tet], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return mean_auc
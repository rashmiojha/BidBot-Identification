from features import User

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def testModel():
    User.basicCountsPerUser()
    User.bidsOnSelf()
    tempTrainData = User.train_data
    X_train = tempTrainData.drop(["bidder_id", "outcome", "payment_account", "address"], axis=1)
    Y_train = tempTrainData["outcome"]
    logisticRegr = LogisticRegression()
    print (cross_val_score(logisticRegr, X_train, Y_train, cv=3).mean())


testModel()

def roc_auc():
    tempTrainData = User.train_data
    tempTestData = User.test_data
    X_train = tempTrainData.drop(["bidder_id", "outcome", "payment_account", "address"], axis=1)
    # X_train = tempTrainData["nb0fBids"].reshape(-1, 1)
    Y_train = tempTrainData["outcome"]
    X = np.array(X_train)
    y = np.array(Y_train)
    cv = StratifiedKFold(y, n_folds=4)
    # classifier = LogisticRegression()
    classifier = RandomForestClassifier(n_estimators=2000, max_depth=20, min_samples_leaf=1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (tran, tet) in enumerate(cv):
        probas_ = classifier.fit(X[tran], y[tran]).predict_proba(X[tet])
        # Compute ROC curve and area the curve
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

    rand = RandomForestClassifier(n_estimators=2000, max_depth=25, min_samples_leaf=1)
    # rand = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01)
    # rand = LogisticRegression()
    rand.fit(X_train[:1020], Y_train[:1020])
    prediction = rand.predict_proba(X_train[1020:])
    prediction_train = rand.predict_proba(X_train[:1020])
    # print (rand.feature_importances_)
    print (metrics.roc_auc_score(Y_train[1020:], prediction[:, 1]))

def submission():
    test = User.test_data
    tempTrainData = User.train_data
    X_train = tempTrainData.drop(["bidder_id", "outcome", "payment_account", "address"], axis=1)
    Y_train = tempTrainData["outcome"]
    X_test = test.drop(['bidder_id', 'payment_account', 'address'], axis=1)
    rand = RandomForestClassifier(n_estimators=1000, max_depth=25, min_samples_leaf=1)
    # print cross_val_score(svm,X_train,y_train,cv=10).mean()
    rand.fit(X_train, Y_train)
    prediction = rand.predict_proba(X_test)
    test['prediction'] = prediction[:, 1]
    test[['bidder_id', 'prediction']].to_csv('submission.csv', index=False)

roc_auc()

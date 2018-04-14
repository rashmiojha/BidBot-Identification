from features import User, User2, Auction, Miscellaneous
from DataProcess import Data
from graphs import roc_auc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def load_features():
    print ("Loading train features")
    train_feature_pkl = open('model/train_features.pkl', 'rb')
    train_features = pickle.load(train_feature_pkl)
    print ("Loaded train features")

    print ("Loading test features")
    test_feature_pkl = open('model/test_features.pkl', 'rb')
    test_features = pickle.load(test_feature_pkl)
    print ("Loaded test features")

    return train_features, test_features

def logistic_regr():
    train_features, test_features = load_features()
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training logistic regression model")
    logisticRegr = LogisticRegression()
    print ("Model trained")
    print ("Cross validation score (Logistic Regression : ")
    cv_score = np.mean(cross_val_score(logisticRegr, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    logisticRegr.fit_transform(X_train, Y_train)
    prediction = logisticRegr.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, logisticRegr)
    print ("AUC score : "+str(auc))

def random_forest():
    train_features, test_features = load_features()
    X_train = train_features.drop(["bidder_id", "outcome"], axis=1)
    Y_train = train_features["outcome"]
    X_test = test_features.drop(["bidder_id"], axis=1)
    print ("Training random forest model")
    randomForest = RandomForestClassifier(n_estimators=2000, max_depth=20, min_samples_leaf=1)
    print ("Model trained")
    print ("Cross validation score (Random Forest) : ")
    cv_score = np.mean(cross_val_score(randomForest, X_train, Y_train, cv=5, scoring='roc_auc'))
    print (cv_score)

    print ("Generating submission file")
    randomForest.fit(X_train, Y_train)
    prediction = randomForest.predict_proba(X_test)
    test_features['prediction'] = prediction[:, 1]
    test_features[['bidder_id', 'prediction']].to_csv('data/submission.csv', index=False)
    print ("Output file successfully created")

    print ("Generating auc curve and auc score")
    auc = roc_auc(train_features, randomForest)
    print ("AUC score : " + str(auc))

def create_and_save():
    print ("Loading data...")
    data = Data('data')
    print ("Extracting features...")
    # print ("1. Extracting basic counts per user")
    # data.train_data, data.test_data = User.basicCountsPerUser(data)
    # print ("2. Extracting basic unique counts per user")
    # data.train_data, data.test_data = User2.basicUniqueCountsPerUser(data)
    # print ("3. Extracting granular merchandise")
    # data.train_data, data.test_data = User2.granularMerchandise(data)
    # print ("4. Extracting bids on self")
    # data.train_data, data.test_data = User.bidsOnSelf(data)
    # print ("5. Extracting auction features")
    # data.train_data, data.test_data = Auction.findAuctionFeatures(data)
    print ("6. Extracting miscellaneous features")
    data.train_data, data.test_data = Miscellaneous.findMiscellaneousFeatures(data)
    print ("Saving train features")
    train_features = data.train_data.drop(["payment_account", "address"], axis=1)
    feature_pkl_filename = 'model/train_features.pkl'
    feature_pkl = open(feature_pkl_filename, 'wb')
    pickle.dump(train_features, feature_pkl)
    feature_pkl.close()
    print ("Train Features saved")
    print ("Saving test features")
    test_features = data.test_data.drop(["payment_account", "address"], axis=1)
    feature_pkl_filename = 'model/test_features.pkl'
    feature_pkl = open(feature_pkl_filename, 'wb')
    pickle.dump(test_features, feature_pkl)
    feature_pkl.close()
    print ("Test Features saved")

def predict_score(algo):
    options = {
        1 : logistic_regr,
        2 : random_forest,
    }
    options[algo]()

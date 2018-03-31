from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from DataProcess import Data
from features import User


def testModel():
    User.basicCountsPerUser()
    tempTrainData = User.train_data
    X_train = tempTrainData.drop(["bidder_id", "outcome", "payment_account", "address"], axis=1)
    Y_train = tempTrainData["outcome"]
    logisticRegr = LogisticRegression()
    print (cross_val_score(logisticRegr, X_train, Y_train, cv=3).mean())


testModel()

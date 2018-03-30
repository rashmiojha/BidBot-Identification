from DataProcess import Data

bid_data = Data('../data').bidData
train_data = Data('../data').trainData

def numberofActions(line, dataGrouped, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]

def findMoreFeatures():
    grBidCount = bid_data['bid_id'].groupby(bid_data['bidder_id']).count()
    return grBidCount

bidCount = findMoreFeatures()
bidderList = bid_data['bidder_id'].unique()
train_data['nb0fBids'] = train_data.apply(lambda x: numberofActions(x, bidCount, bidderList), axis=1)
print (train_data)
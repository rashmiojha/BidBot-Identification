from DataProcess import Data

bid_data = Data('../data').bidData
train_data = Data('../data').trainData
test_data = Data('../data').testData

def numberofActions(line, dataGrouped, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]

def findMoreFeatures():
    print ("")
    # grBidCount = bid_data['bid_id'].groupby(bid_data['bidder_id']).count()
    # grMerchandiseCount = bid_data['merchandise'].groupby(bid_data['bidder_id']).count()
    # payAccCount = train_data['payment_account'].groupby(train_data['bidder_id']).count()
    # addressCount = train_data['address'].groupby(train_data['bidder_id']).count()
    #
    # bidderList = bid_data['bidder_id'].unique()
    #
    # train_data['nb0fBids'] = train_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)
    # test_data['nb0fBids'] = test_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)
    #
    # train_data['nb0fMerch'] = train_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)
    # test_data['nb0fMerch'] = test_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)
    #
    # train_data['nb0fPayAcc'] = train_data.apply(lambda x: numberofActions(x, payAccCount, bidderList), axis=1)
    # test_data['nb0fPayAcc'] = test_data.apply(lambda x: numberofActions(x, payAccCount, bidderList), axis=1)
    #
    # train_data['nb0fAdress'] = train_data.apply(lambda x: numberofActions(x, addressCount, bidderList), axis=1)
    # test_data['nb0fAdress'] = test_data.apply(lambda x: numberofActions(x, addressCount, bidderList), axis=1)

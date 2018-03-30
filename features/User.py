from DataProcess import Data

bid_data = Data('../data').bidData
train_data = Data('../data').trainData
test_data = Data('../data').testData

print (bid_data)

def numberofActions(line, dataGrouped, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]

def basicCountsPerUser():
    bidderList = bid_data['bidder_id'].unique()
    countryCount = bid_data['country'].groupby(bid_data['bidder_id']).count()
    ipCount = bid_data['ip'].groupby(bid_data['bidder_id']).count()
    urlCount = bid_data['url'].groupby(bid_data['bidder_id']).count()
    deviceCount = bid_data['device'].groupby(bid_data['bidder_id']).count()
    auctionCount = bid_data['auction'].groupby(bid_data['bidder_id']).count()
    grBidCount = bid_data['bid_id'].groupby(bid_data['bidder_id']).count()
    grMerchandiseCount = bid_data['merchandise'].groupby(bid_data['bidder_id']).count()
    payAccCount = train_data['payment_account'].groupby(train_data['bidder_id']).count()
    addressCount = train_data['address'].groupby(train_data['bidder_id']).count()

    train_data['nb0fCountry'] = train_data.apply(lambda x: numberofActions(x, countryCount, bidderList), axis=1)
    test_data['nb0fCountry'] = test_data.apply(lambda x: numberofActions(x, countryCount, bidderList), axis=1)

    train_data['nb0fIP'] = train_data.apply(lambda x: numberofActions(x, ipCount, bidderList), axis=1)
    test_data['nb0fIP'] = test_data.apply(lambda x: numberofActions(x, ipCount, bidderList), axis=1)

    train_data['nb0fURL'] = train_data.apply(lambda x: numberofActions(x, urlCount, bidderList), axis=1)
    test_data['nb0fURL'] = test_data.apply(lambda x: numberofActions(x, urlCount, bidderList), axis=1)

    train_data['nb0fDevice'] = train_data.apply(lambda x: numberofActions(x, deviceCount, bidderList), axis=1)
    test_data['nb0fDevice'] = test_data.apply(lambda x: numberofActions(x, deviceCount, bidderList), axis=1)

    train_data['nb0fAuction'] = train_data.apply(lambda x: numberofActions(x, auctionCount, bidderList), axis=1)
    test_data['nb0fAuction'] = test_data.apply(lambda x: numberofActions(x, auctionCount, bidderList), axis=1)

    train_data['nb0fBids'] = train_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)
    test_data['nb0fBids'] = test_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)

    train_data['nb0fMerch'] = train_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)
    test_data['nb0fMerch'] = test_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)

    train_data['nb0fPayAcc'] = train_data.apply(lambda x: numberofActions(x, payAccCount, bidderList), axis=1)
    test_data['nb0fPayAcc'] = test_data.apply(lambda x: numberofActions(x, payAccCount, bidderList), axis=1)

    train_data['nb0fAdress'] = train_data.apply(lambda x: numberofActions(x, addressCount, bidderList), axis=1)
    test_data['nb0fAdress'] = test_data.apply(lambda x: numberofActions(x, addressCount, bidderList), axis=1)

basicCountsPerUser()

print (train_data)
print (test_data)
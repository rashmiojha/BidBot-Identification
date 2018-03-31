from DataProcess import Data
import pandas as pd
import numpy as np
bid_data = Data('../data').bidData
dict = {'jewelry': 0, 'furniture': 1, 'home goods': 2, 'mobile': 3, 'sporting goods': 4,
         'office equipment': 5, 'computers': 6, 'books and music': 7, 'clothing': 8,
         'auto parts': 9}
def numberofActions(line, dataGrouped, dataid):
    if not line['bidder_id'] in dataid:
        return 0
    else:
        return dataGrouped[line['bidder_id']]

def findMerchandise(line,grouped,dict,dataid):
    res = np.zeros(10)
    if line in dataid:
        merch = np.array(grouped[grouped['bidder_id']==line]['merchandise'])[0]
        print merch
        res[dict[merch]] = 1
    return tuple(res)

def basicUniqueCountsPerUser(train_data,test_data):
    bidderList = bid_data['bidder_id'].unique()
    countryCount = bid_data['country'].groupby(bid_data['bidder_id']).nunique()
    ipCount = bid_data['ip'].groupby(bid_data['bidder_id']).nunique()
    urlCount = bid_data['url'].groupby(bid_data['bidder_id']).nunique()
    deviceCount = bid_data['device'].groupby(bid_data['bidder_id']).nunique()
    auctionCount = bid_data['auction'].groupby(bid_data['bidder_id']).nunique()
    grBidCount = bid_data['bid_id'].groupby(bid_data['bidder_id']).nunique()
    grMerchandiseCount = bid_data['merchandise'].groupby(bid_data['bidder_id']).nunique()
    payAccCount = train_data['payment_account'].groupby(train_data['bidder_id']).nunique()
    addressCount = train_data['address'].groupby(train_data['bidder_id']).nunique()

    train_data['nb0fUniqueCountry'] = train_data.apply(lambda x: numberofActions(x, countryCount, bidderList), axis=1)
    test_data['nb0fUniqueCountry'] = test_data.apply(lambda x: numberofActions(x, countryCount, bidderList), axis=1)

    train_data['nb0fUniqueIP'] = train_data.apply(lambda x: numberofActions(x, ipCount, bidderList), axis=1)
    test_data['nb0fUniqueIP'] = test_data.apply(lambda x: numberofActions(x, ipCount, bidderList), axis=1)

    train_data['nb0fUniqueURL'] = train_data.apply(lambda x: numberofActions(x, urlCount, bidderList), axis=1)
    test_data['nb0fUniqueURL'] = test_data.apply(lambda x: numberofActions(x, urlCount, bidderList), axis=1)

    train_data['nb0fUniqueDevice'] = train_data.apply(lambda x: numberofActions(x, deviceCount, bidderList), axis=1)
    test_data['nb0fUniqueDevice'] = test_data.apply(lambda x: numberofActions(x, deviceCount, bidderList), axis=1)

    train_data['nb0fUniqueAuction'] = train_data.apply(lambda x: numberofActions(x, auctionCount, bidderList), axis=1)
    test_data['nb0fUniqueAuction'] = test_data.apply(lambda x: numberofActions(x, auctionCount, bidderList), axis=1)

    train_data['nb0fUniqueBids'] = train_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)
    test_data['nb0fUniqueBids'] = test_data.apply(lambda x: numberofActions(x, grBidCount, bidderList), axis=1)

    train_data['nb0fUniqueMerch'] = train_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)
    test_data['nb0fUniqueMerch'] = test_data.apply(lambda x: numberofActions(x, grMerchandiseCount, bidderList), axis=1)

    train_data['nb0fUniquePayAcc'] = train_data.apply(lambda x: numberofActions(x, payAccCount, bidderList), axis=1)
    test_data['nb0fUniquePayAcc'] = test_data.apply(lambda x: numberofActions(x, payAccCount, bidderList), axis=1)

    train_data['nb0fUniqueAdress'] = train_data.apply(lambda x: numberofActions(x, addressCount, bidderList), axis=1)
    test_data['nb0fUniqueAdress'] = test_data.apply(lambda x: numberofActions(x, addressCount, bidderList), axis=1)

def granularMerchandise(train_data,test_data):
    merchList = bid_data['merchandise'].unique()
    print merchList
    train_data[['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
           'office equipment', 'computers', 'books and music', 'clothing',
           'auto parts']] = pd.DataFrame(np.zeros((train_data.shape[0], 10)),
                                         columns=['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
                                                  'office equipment', 'computers', 'books and music', 'clothing',
                                                  'auto parts'])

    test_data[['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
          'office equipment', 'computers', 'books and music', 'clothing',
          'auto parts']] = pd.DataFrame(np.zeros((train_data.shape[0], 10)),
                                        columns=['jewelry', 'furniture', 'home goods', 'mobile', 'sporting goods',
                                                 'office equipment', 'computers', 'books and music', 'clothing',
                                                 'auto parts'])

    grouped = bid_data[['bidder_id', 'merchandise']].drop_duplicates()
    res = train_data['bidder_id'].map(lambda x: findMerchandise(x, grouped, dict, merchList))
    (train_data['jewelry'], train_data['furniture'], train_data['home goods'],
     train_data['mobile'], train_data['sporting goods'], train_data['office equipment'],
     train_data['computers'], train_data['books and music'], train_data['clothing'], train_data['auto parts']) = zip(*res)
    res = test_data['bidder_id'].map(lambda x: findMerchandise(x, grouped, dict, merchList))
    (test_data['jewelry'], test_data['furniture'], test_data['home goods'],
     test_data['mobile'], test_data['sporting goods'], test_data['office equipment'],
     test_data['computers'], test_data['books and music'], test_data['clothing'], test_data['auto parts']) = zip(*res)
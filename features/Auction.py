from DataProcess import Data

bid_data = Data('../data').bidData
train_data = Data('../data').trainData

def findAuctionFeatures():
    auctionList = bid_data['auction'].unique()
    grCountryCount = bid_data['country'].groupby(bid_data['auction']).count()
    train_data['nbCountryPa'] = grCountryCount
    print (train_data)
findAuctionFeatures()


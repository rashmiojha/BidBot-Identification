from DataProcess import Data

bid_data = Data('../data').bidData
print (bid_data)

def findAuctionFeatures():
    auctionList = bid_data['auction'].unique()
    for row in bid_data:
        print (row)



import matplotlib.pyplot as plt;  plt.rcdefaults()
from DataProcess import Data
import pandas as pd
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

allGraphs()
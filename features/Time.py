from DataProcess import Data
import pandas as pd

def timeStartEndDiff(data):
    bid_data = data.bidData
    train_data = data.trainData
    test_data = data.testData

    start_time = bid_data.groupby('auction').time.min().reset_index()
    start_time = start_time.rename(columns={'time': 'start'})
    end_time = bid_data.groupby('auction').time.max().reset_index()
    end_time = end_time.rename(columns={'time': 'end'})

    times = pd.merge(start_time, end_time, on='auction', how='left')
    bid_data = pd.merge(bid_data, times[['auction', 'start', 'end']], on='auction', how='left')

    bid_data['endDiff'] = bid_data.end - bid_data.time
    bid_data['startDiff'] = bid_data.time - bid_data.start

    b = bid_data.groupby('bidder_id').endDiff.median().reset_index()
    b = b.rename(columns={'endDiff': 'endDiffMedian'})
    train_data = pd.merge(train_data, b, on='bidder_id', how='left')
    test_data = pd.merge(test_data, b, on='bidder_id', how='left')

    b = bid_data.groupby('bidder_id').startDiff.median().reset_index()
    b = b.rename(columns={'startDiff': 'startDiffMedian'})
    train_data = pd.merge(train_data, b, on='bidder_id', how='left')
    test_data = pd.merge(test_data, b, on='bidder_id', how='left')

    return train_data, test_data
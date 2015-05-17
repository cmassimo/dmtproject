import pandas as pd
import numpy as np
from math import log, floor
import sklearn as sk
from sklearn import preprocessing

def sample_dataset(ds, exclude_srch_ids):

    # getting the unique values for srch_id
    ids = ds['srch_id'].unique() - exclude_srch_ids

    cutoff = floor(ids.size*0.1)

    # sample 1000 srch_id(s)
    smpl = np.random.choice(ids, cutoff, False)

    # filter out the dataset
    new_ds = ds[ds['srch_id'].isin(smpl)]

    # save it to csv
    new_ds.to_csv(os.path.join('..', 'data', 'validation.csv'))

    return new_ds

def oversample_dataset(ds, outname, save_csv=True):

    # getting the values for prop_id
    bids = ds[ds['booking_bool']==1].index.values
    cids = ds[ds['booking_bool']==0][ds['click_bool']==1].index.values
    nids = ds[ds['booking_bool']==0][ds['click_bool']==0].index.values

    bcutoff = bids.size
    ncutoff = int(floor(bcutoff*0.2))

    bsmpl = bids
    csmpl = cids
    nsmpl = np.random.choice(nids, ncutoff, False)

    # filter out the dataset
    rows = pd.concat([ds[ds.index.isin(bsmpl)], ds[ds.index.isin(csmpl)], ds[ds.index.isin(nsmpl)]])

    # save it to csv?
    if save_csv:
        rows.to_csv(outname)

    return rows

def oversampled_low_price_dataset(inname, outname, price, save_csv=True):
    '''
    price = threshold of prices to be considered ( < price)
    '''
    #Opening the datafile
    ds = pd.read_csv(inname)

    # getting the unique values for prop_id
    bids = ds[(ds['booking_bool'] == 1) & (ds['price_usd'] < price)].drop_duplicates('prop_id').index.values
    nbids = ds[(ds['booking_bool'] == 0) & (ds['price_usd'] < price)].drop_duplicates('prop_id').index.values

    # sample 30000 records (50-50%)
    bsmpl = np.random.choice(bids, 15000, False)
    nbsmpl = np.random.choice(nbids, 15000, False)

    # filter out the dataset
    rows = pd.concat([ds[ds.index.isin(bsmpl)], ds[ds.index.isin(nbsmpl)]])

    # save it to csv?
    if save_csv:
        rows.to_csv(outname)

    return rows

def norm_pcid(dset, key):
    res = dset[key]. \
        groupby(dset['prop_country_id']). \
        apply(lambda x: (x - x.mean()) / x.std())
    return res

def log_norm_srch_id(dataset, key):
    ''' dataset = pandas dataframe
        key = index of variable to be normalized (string)
        Returns normalized log of column by srch_id.
    '''
    tmp_log = dataset[key].apply(lambda x: log(x+1))
    #normalize by each srch_id
    slog = tmp_log.groupby(dataset['srch_id']).apply(lambda x: (x-x.mean())/x.std())

    return slog

def loc_ratio2(dset):
    nlog_price = log_norm_srch_id(dset, 'price_usd')
    nlog_price_center = nlog_price.groupby(dset['srch_id']).apply(lambda x: (x - min(x))/(max(x) - min(x))+1)
    return dset['prop_location_score2'].fillna(0) / nlog_price_center

def label(dset):
   return dset['booking_bool'] + dset['click_bool']

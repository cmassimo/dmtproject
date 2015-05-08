import pandas as pd
import numpy as np

# DEPRECATED
def sample_dataset(inname, outname):
#    print('DEPRECATED FUNCTION: use "oversampled_dataset(inname, outname, save_csv =True)" instead.')
#    return None

    #Opening the datafile
    ds = pd.read_csv(inname)

    # getting the unique values for srch_id
    ids = ds['srch_id'].unique()

    #ids.size

    # sample 1000 srch_id(s)
    smpl = np.random.choice(ids, 1000, False)

    # filter out the dataset
    new_ds = ds[ds['srch_id'].isin(smpl)]

    # save it to csv
    if (save_csv):
        new_ds.to_csv(outname)

    return new_ds

def oversampled_dataset(inname, outname, save_csv=True):

    #Opening the datafile
    ds = pd.read_csv(inname)

    # getting the unique values for prop_id
    bids = ds[ds['booking_bool']==1].drop_duplicates('prop_id').index.values
    nbids = ds[ds['booking_bool']==0].drop_duplicates('prop_id').index.values

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

#ds['norm_review_rating'] = norm_pcid(ds, 'prop_starrating')
#
#ds[['norm_review_rating', 'booking_bool']].corr()
#
#ds['srch_comp'] = ds['srch_length_of_stay'] / (ds['srch_adults_count'] + 0.5*ds['srch_room_count'] + 1.5*ds['srch_children_count'])

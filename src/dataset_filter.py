import pandas as pd
import numpy as np

def sample_dataset(inname, outname):
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
    new_ds.to_csv(outname)

    return new_ds
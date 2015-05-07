import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#for loading smaller dataset
sys.path.append('C:\Users\david\Dropbox\Data Mining Techniques\Assignment2')

from dataset_filter import sample_dataset

#load the entire dataset
#ds = pd.read_csv('training_set_VU_DM_2014.csv')

#load a smaller dataset
ds = sample_dataset('training_set_VU_DM_2014.csv', 'small_dataset.csv')

def STATAclean(ds):
    STATAds = ds
    for num in range(1,9):
        STATAds = STATAds.drop('comp'+str(num)+'_rate', 1).drop('comp'+str(num)+'_inv', 1).drop('comp'+str(num)+'_rate_percent_diff', 1)
    
    STATAds = STATAds.replace(to_replace='NaN', value='.')
    
    STATAds.to_csv('STATA_small_dataset.csv', sep=',')

    print "New STATA file created"

#############################
#first 5 entries
ds.head()

#filter stuff
randomfilter = ds[ds['random_bool'] == 1]

#filter is like a subset of the dataset now
randomfilter['booking_bool'].mean()

orderedfilter = ds[ds['random_bool'] == 0]
orderedfilter['booking_bool'].mean()

#count proportion of bookings -> need to count the occurences of a particular value before plotting
ds['booking_bool'].value_counts()

#plot that 
ds['booking_bool'].value_counts().plot(kind='pie', figsize=(5,5), labels=['not booked', 'booked'])

#review scores
ds['prop_review_score'].value_counts()
ds['prop_review_score'].value_counts().plot(kind='pie', figsize=(5,5))

#star ratings
ds['prop_starrating'].value_counts().plot(kind='pie', figsize=(5,5))
ds['prop_starrating'].plot(kind='hist', bins=5, figsize=(5,5))


    
    
    

# coding: utf-8

# In[ ]:


#!/usr/bin/python2.7

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from check_files import path_check
import time

def change_date(df):
    date_changed = df
    date_changed['date'] = map(lambda x: datetime.strptime(str(x), '%Y%m%d%H'),
                               date_changed.date)
    return date_changed

def flatten_data(df, name):
    """Make all samples with the same timestamp on the same line."""
    
    df = df.drop(name, axis=1)
    return df.reset_index(drop=True).T

def pre_clean_data():
    """
    pre_clean the raw data and put the pre_cleaned data into a new folder.
    Data is indexed by date, which has a presentation of 2012-04-28 11:00:00.
    May take up to 3m 30s for 7 windfarms.

    """

    default_raw_dir = './data/raw/'
    raw_directory = raw_input("Enter raw data directory {0}): ".
                              format(default_raw_dir)) or default_raw_dir
#     raw_directory = default_raw_dir

    default_pre_cleaned_dir = './data/pre_cleaned/'
    pre_cleaned_directory = raw_input("Enter pre_cleaned data directory ({0}):".
                                  format(default_pre_cleaned_dir)) or default_pre_cleaned_dir
#     pre_cleaned_directory = default_pre_cleaned_dir
    
    path_check(pre_cleaned_directory);
    
    start = time.time()
    
    train = pd.read_csv(raw_directory+'train.csv',sep = ',')
    
    # Change the timestamp presentation and set index.
    train = change_date(train)
    train = train.set_index('date')

    benchmark = pd.read_csv(raw_directory+'benchmark.csv',sep = ',')
    benchmark = change_date(benchmark)
    benchmark = benchmark.set_index('date')

    # Process 7 windfarms' data.
    for i in range(1,8):
        wf = pd.read_csv(raw_directory+'windforecasts_wf{0}.csv'.format(i),sep = ',')
        
        wf = change_date(wf)

        # Add hors to the timestamp, so that it represents the actual forcast time.
        wf['date'] = map(lambda x, y: x + timedelta(hours = y), wf.date, wf.hors)
        
        # Flatten the multiindex in the column and get single column key.
        wf = wf.groupby('date').apply(flatten_data, 'date').unstack()
        wf.columns = wf.columns.values
        
        #Output pre_cleaned files.
        train_wf = pd.merge(wf,pd.DataFrame((train["wp{0}".format(i)])), 
                 how='inner', left_index= True, right_index= True)
        train_wf.to_csv(pre_cleaned_directory+"train_wf{0}.csv".format(i),sep = ',')
        
        benchmark_wf = pd.merge(wf,pd.DataFrame((benchmark["wp{0}".format(i)])),
                 how='inner', left_index= True, right_index= True)
        benchmark_wf.to_csv(pre_cleaned_directory+"benchmark_wf{0}.csv".format(i),sep = ',')
        
        print "Output pre_cleaned data for wf{0}.".format(i)

    end = time.time()
    elapsed = end -start
    minutes, seconds = divmod(elapsed, 60)
    
    print 'Done. Time = {:0>1}m {:05.2f}s.'.format(int(minutes), seconds)
    
    
if __name__ == '__main__':
    
    pre_clean_data()


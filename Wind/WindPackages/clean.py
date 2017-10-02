
# coding: utf-8

# In[ ]:


#!/usr/bin/python2.7

import numpy as np
import pandas as pd
import time as time
from check_files import path_check
from datetime import datetime
from math import isnan

def post_clean_data():
    """
    Change pre_cleaned data into cleaned pure-number data 
    (without head and timestamp) that can be used for the
    training. The output dataset is NOT normalized.
    
    Each train dataset has 18756 rows.
    Each benchmark dataset has 7488 rows.
    Each dataset has 17 columns.
    First 4 columns for the 1st forecast of u,v, ws, wd, 
    second 4 columns for the 2nd forecast,...,
    the last for the normalized wind power(label).
    
    Data example:
    | u(1) v(1) ws(1) wd(1) | u(2) v(2) ws(2) wd(2) | u(3) v(3) ws(3) wd(3)| u(4) v(4) ws(4) wd(4) | power |
  
    NAN values in the pre_cleaned dataset is set to be the average
    value of the known forecast.
    
    Example: For the train dataset of wf1 at 2009-07-01 13:00:00,
    we only have two preditions from 2009-07-01 00:00:00 at hor 1 
    (the 1st prediction) and 2009-07-01 12:00:00 at hor 1 (the 2nd 
    prediction), then, the 3rd, 4th predition values of u, v, ws,
    wd are set to be the average value of the 1st and 2nd predictions.

    """

    default_pre_cleaned_dir = './data/pre_cleaned/'
    pre_cleaned_directory = raw_input("Enter pre_cleaned data directory {0}): ".
                              format(default_pre_cleaned_dir)) or default_pre_cleaned_dir
    # pre_cleaned_dir= default_pre_cleaned_dir

    default_cleaned_dir = './data/cleaned/'
    cleaned_directory = raw_input("Enter cleaned data directory ({0}):".
                                  format(default_cleaned_dir)) or default_cleaned_dir
    # cleaned_directory = default_cleaned_dir

    path_check(cleaned_directory);
    
    wf_num = 7
    prefix = ['train','benchmark']
    
    start = time.time()
    
    for num in range(1,wf_num+1):
        for pre in prefix:
            wf = pd.read_csv(pre_cleaned_directory+'{0}_wf{1}.csv'.format(pre, num),sep = ',')
            wf = wf.values
            
            # Delete timestamp.
            wf = np.delete(wf,0,1)
            
            #Delete hor.
            for i in range(4):
                wf = np.delete(wf, 4*i, 1)
                
            # Do the average of previous predictions 
            # if the value if NAN.
            for row in range(wf.shape[0]):
                for col in range(wf.shape[1]):
                    if isnan(wf[row][col]):
                        s = 0
                        p = col/4
                        r = col%4
                        for i in range(p):
                            s += wf[row][r + i*4]
                            wf[row][col] = s/p
            np.savetxt(cleaned_directory+'{0}_wf{1}.csv'.format(pre, num), wf, fmt='%.3f', delimiter=",")
            print '{0}_wf{1}.csv is cleaned.'.format(pre, num)
            
    end = time.time()
    elapsed = end -start
    minutes, seconds = divmod(elapsed, 60)
    
    print 'Done. Time = {:0>1}m {:05.2f}s.'.format(int(minutes), seconds)

    
if __name__ == '__main__':
    
    post_clean_data()


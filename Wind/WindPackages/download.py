
# coding: utf-8

# In[ ]:


#!/usr/bin/python2.7

import requests
import getpass
from check_files import path_check
import time

def download_data():
    """ 
    Download data from Kaggle. 
    Need Kaggle ID, password, competition name, files, and local directory.
    No check of the correctness.
    Values in the parentheses are the default values.
    No protection from '429 Too Many Requests' error. 
    Default chunk size: 1 MB.
    
    Examples:
    Enter Kaggle username (None): UserName
    Enter Kaggle password (None): ········
    Enter Kaggle competition (GEF2012-wind-forecasting): 
    Enter Kaggle files (['train.csv', 'benchmark.csv', 'windforecasts_wf1.csv',
    'windforecasts_wf2.csv', 'windforecasts_wf3.csv', 'windforecasts_wf4.csv',
    'windforecasts_wf5.csv', 'windforecasts_wf6.csv', 'windforecasts_wf7.csv']):
    Enter local directory (./data/raw/):  
    
    """

    Default = {
        'UserName': None,
        'Password': None,
        'Competition': 'GEF2012-wind-forecasting',
        'Files': ['train.csv', 
                  'benchmark.csv',
                  'windforecasts_wf1.csv',
                  'windforecasts_wf2.csv',
                  'windforecasts_wf3.csv',
                  'windforecasts_wf4.csv',
                  'windforecasts_wf5.csv',
                  'windforecasts_wf6.csv',
                  'windforecasts_wf7.csv',                  
                 ],
        'Dir': './data/raw/'
    }
    
    username = raw_input("Enter Kaggle username ({0}): ".
                         format(Default['UserName'])) or Default['UserName']
    pwd = getpass.getpass("Enter Kaggle password ({0}): ".
                          format(Default['Password'])) or Default['Password']
    competition = raw_input("Enter Kaggle competition ({0}): ".
                            format(Default['Competition'])) or Default['Competition']
    files = raw_input("Enter Kaggle files ({0}): ".
                      format(Default['Files'])).split() or Default['Files']
    directory = raw_input("Enter local directory ({0}): ".
                          format(Default['Dir'])) or Default['Dir']
        
    kaggle_info = {}
    kaggle_info['UserName'] = username
    kaggle_info['Password'] = pwd
    
    path_check(directory);
    
#     links = re.findall(
#             '"url":"(/c/{}/download/[^"]+)"'.format(competition), data
#         )
    start = time.time()

    for i in range(len(files)):
        
        filename = files[i]
        local_filename = files[i]
        
#         data_url = 'https://www.kaggle.com/c/{0}/download/{1}'.format(competition, filename)
#         r = requests.get(data_url)
#         file_url = r.url
        
        file_url = 'https://www.kaggle.com/account/login?ReturnUrl=%2fc%2f{0}%2fdownload%2f{1}'.format(competition, filename)       
        
        r = requests.post(file_url, data = kaggle_info, stream = False)

        f = open(directory+local_filename, 'w')
        
        for chunk in r.iter_content(chunk_size = 1024 * 1024):
            if chunk:
                f.write(chunk)
                
        f.close()
        
        print '{0} fetched.'.format(filename)
        
    end = time.time()
    elapsed = end -start
    minutes, seconds = divmod(elapsed, 60)
    
    print 'Done. Time = {:0>1}m {:05.2f}s.'.format(int(minutes), seconds)
    
if __name__ == '__main__':
    
    download_data()


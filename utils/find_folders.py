"""EXPLANATION OF SCIPRT CONTENT"""

import os
import pandas as pd
import numpy as np
import filecmp

def get_onedrive_path(
    folder: str = 'onedrive',
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored
    Folder has to be in ['onedrive', 'Percept_Data_structured', 'sourcedata']
    """
    folder_options = [
        'onedrive', "ftg", 'data', 'results', 'figures'
    ]
    # Error checking, if folder input is in folder options
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be in {folder_options}')

    # from your cwd get the path and stop at 'Users'
    path = os.getcwd()
    if path[0] == "T":
        
        path = os.path.join("C:", "Users", "mathiopv")

    elif path[0] == "C":
        
        while os.path.dirname(path)[-5:] != 'Users':
            path = os.path.dirname(path) # path is now leading to Users/username
    
    # get the onedrive folder containing "onedrive" and "charit" and add it to the path
    onedrive_f = [
        f for f in os.listdir(path) if logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower())
    ]
    path = os.path.join(path, onedrive_f[0]) # path is now leading to Onedrive folder

    if folder.lower() == "onedrive": return path

    elif folder.lower() == "ftg": return os.path.join(path, "FTG_PROJECT")
    
    elif folder.lower() == "data":
        return os.path.join(path, "FTG_PROJECT", 'data')
    
    elif folder.lower() == "figures":
        return os.path.join(path, "FTG_PROJECT", 'figures')
    
    elif folder.lower() == "results":
        return os.path.join(path, "FTG_PROJECT", 'results')


ftg_path = get_onedrive_path("FTG")

pat_path = os.path.join(
    os.path.join(
        ftg_path,
        'data',
        'raw_data',
        'raw_mats',
        "sub029"
    )
)

metadata = pd.read_excel(
    os.path.join(
        pat_path,
        "sub029_metadata.xlsx"
    ),
    sheet_name='Sheet1'
)


mat_files =  pd.Series(
    [f for f in os.listdir(pat_path) if f.endswith('.mat')])


id = metadata['perceiveFilename']

for jk in np.arange(0,len(mat_files)):
    idx = metadata.idx[metadata.perceiveFilename == mat_files[jk]]
    print(idx[0])
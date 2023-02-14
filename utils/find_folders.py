"""EXPLANATION OF SCIPRT CONTENT"""

import os
from numpy import logical_and

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
    if path[0] == "t":
        
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
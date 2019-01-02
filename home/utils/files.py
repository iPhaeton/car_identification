import numpy as np

def filter_files(files_list, filter_by = None):
    try:
        files_list.remove('.DS_Store')
    except:
        x=1
        
    if filter_by is not None:
        mask = np.logical_not(np.isin(files_list, filter_by))
        result = []
        for f, m in zip(files_list, mask):
            if m == True:
                result.append(f)

        return result
    else:
        return files_list
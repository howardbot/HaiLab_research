from scipy.io import loadmat
import numpy as np
#Loading .mat file and return TInfo_new structure
def load_mat_session(filepath):
    mat = loadmat(filepath,struct_as_record=False, squeeze_me = True)
    T = mat['TInfo_new']

    #if there is only one trial, then convert it to list
    if isinstance(T,np.ndarray):
        return  T.tolist()
    # will be converted to an object then has _dict_ property
    elif hasattr(T,'_dict_'):
        return [T]
    else:
        raise ValueError("Unexpected structure in TInfo_new")
from braindecode.util import set_random_seeds, np_to_var, var_to_np
import matplotlib.pyplot as plt
from moabb.datasets import PhysionetMI, BNCI2014001
from moabb.paradigms import MotorImagery
from numpy.random import RandomState
import pickle
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import mne

import util.shallow_net
from util.utilfunc import get_balanced_batches
from util.preproc import plot_confusion_matrix

import numpy as np

def matmul_list(matrix_list, db=False):
    if db:
        for x in matrix_list:
            print(x.shape)
    number_matrix = len(matrix_list)
    result = np.copy(matrix_list[0])
    for i in range(1, number_matrix):
        result = np.matmul(result, matrix_list[i])
    return result

cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)



ds_src2 = PhysionetMI()
raw = ds_src2.get_data(subjects=[1])[1]['session_0']['run_1']
channels = raw.pick_types(eeg=True).ch_names
print(channels)
stop

listChns = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 
            'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 
            'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']

fmin, fmax = 15, 32
sfreq = 160.
# prgm_4classes = MotorImagery(n_classes=4, resample=sfreq, fmin=fmin, fmax=fmax)

# X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=[x+1 for x in range(5)])

# np.save('./testPhi', X_src2)

X_src2 = np.load('./testPhi.npy', allow_pickle=True)
print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))


print(X_src2[0].shape)

for ii in range(100):
    allMat = X_src2[ii].T
    meanRow = np.mean(allMat, axis = 1, keepdims = True)
    meanMat = np.matmul(np.ones((allMat.shape[1], 1)), meanRow.T).T

    stdRow = np.mean(allMat, axis = 1, keepdims = True)
    stdMat = np.matmul(np.ones((allMat.shape[1], 1)), meanRow.T).T

    allMat = (allMat - meanRow)/stdMat
    # tmpMat = np.random.rand(20, 18)
    # tmpMat = np.matmul(allMat, allMat.T)
    tmpMat = allMat
    U, S, V = np.linalg.svd(tmpMat, full_matrices=False)
    print(np.sum(np.abs(tmpMat - np.matmul(np.matmul(U, np.diag(S)), V))))
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

# ds_src2 = PhysionetMI()
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
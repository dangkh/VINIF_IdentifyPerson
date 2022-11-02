import matplotlib.pyplot as plt 
import numpy as np
import mne
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from mne.filter import filter_data
import pickle
import os
import sys
from scipy import signal
import time
from sklearn.utils import shuffle
from scipy.linalg import sqrtm

def balance_set(X,y,replace=False):
    labels= np.unique(y)
    labelsize=labels.shape[0]
    #print('labelsize:',labelsize)
    label_count = np.zeros(labelsize).astype(int)
    for i in range(labelsize):
        tempy = y[y==labels[i]]
        label_count[i]=y[y==labels[i]].shape[0]
    maxsize = label_count.max()
    for i in range(labelsize):
        tempy = y[y==labels[i]]
        tempx = X[y==labels[i]]
        tempx,tempy,ratio=get_sourceset_from_newraces(tempx,tempy,maxsize,random=True,replace=replace)
        if i ==0:
            balanced_data = tempx
            balanced_label = tempy
        else:
            balanced_data = np.concatenate((balanced_data,tempx))
            balanced_label = np.concatenate((balanced_label,tempy))
    return shuffle(balanced_data,balanced_label)


def plot_confusion_matrix(y_true, y_pred, classes,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
#     # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')



    # fig, ax = plt.subplots()
    # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # # We want to show all ticks...
    # ax.set(xticks=np.arange(cm.shape[1]),
    #     yticks=np.arange(cm.shape[0]),
    #     # ... and label them with the respective list entries
    #     xticklabels=classes, yticklabels=classes,
    #     title=title,
    #     ylabel='True label',
    #     xlabel='Predicted label')

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #     rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, format(cm[i, j], fmt),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    # plt.show()
    # return ax

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandpass')
    y= signal.lfilter(i, u, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandstop')
    y= signal.lfilter(i, u, data)
    return y


def lowpass_cnt(data,lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    i, u = signal.butter(order, low, btype='lowpass')
    y= signal.lfilter(i, u, data)
    return y

def highpass_cnt(data,highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    i, u = signal.butter(order, high, btype='highpass')
    y= signal.lfilter(i, u, data)
    return y


def filter_notch(data,notchcut,fs):
    f0 = notchcut  # Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2)  # Normalized Frequency
    Q= 30
    i, u = signal.iirnotch(w0, Q)
#     data = signal.filtfilt(b, a, data)
    data= signal.lfilter(i, u, data)
    return(data)

def calculateAdj(inputMat):
    mat = np.asarray(inputMat)
    s1, s2, s3, s4 = mat.shape
    mat = mat.reshape(s1 * s2, s3, s4)
    coMat = mat.mean(axis=0)
    Adj = np.corrcoef(coMat[:, :].T)
    # print(Adj)
    D = np.array(np.sum(Adj, axis=0))
    D = np.sqrt(D)
    # print(np.matrix(np.diag(D)).shape)
    # print(np.diag(D).shape)
    # print(np.diag(D))
    # print(np.matrix(np.diag(D)))

    # print(np.diag(D).shape)
    # print(np.matrix(np.diag(D)).shape)

    D = np.diag(D)
    matAdj = torch.Tensor(np.linalg.inv(D) * Adj * np.linalg.inv(D)).to(device)
    # matAdj = get_adj(Adj)
    return matAdj

def get_adj(adj):
    """
    build symmetric adjacency matrix
    @param adj:
    @return:
    """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def getNormR(data, shapeIn = 32):
    print(data.shape)
    normR = np.zeros([shapeIn, shapeIn])
    for ii in range(len(data)):
        covX = np.matmul(data[ii].T, data[ii])
        normR += covX
    normR = normR / len(data)
    # normR = sqrtm(normR)
    normR = np.linalg.inv(normR)
    return normR


# Calculate FFT of data Pos or Neg
def GetFFT(datas, lenTrial = 128):
    sampling_length = lenTrial
    fft_rs = []
    for data in datas:
        ts = 1.0/ lenTrial

        freq = np.fft.fftfreq(len(data), d = ts)
        fft = np.fft.fft(data)
        fft_rs.append(fft)

    return fft_rs, freq

# Calculate Power Spectral Density(PSD) (equation 1)
def GetPSD(datas, lenTrial = 128):
    sampling_length = lenTrial
    PSD_rs = []
    for data in datas:
        PSD_seg = []
        for d in data:
            PSD_n = d * np.conjugate(d)/lenTrial
            PSD_seg.append(PSD_n)
        PSD_rs.append(PSD_seg)
  
    return PSD_rs

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def MA(datas, windowSize = 4):
    newData = []
    for data in datas:
        tmp = moving_average(data, windowSize)
        newData.append(tmp)

    return np.vstack(newData)


def matmul_list(matrix_list, db=False):
    if db:
        for x in matrix_list:
            print(x.shape)
    number_matrix = len(matrix_list)
    result = np.copy(matrix_list[0])
    for i in range(1, number_matrix):
        result = np.matmul(result, matrix_list[i])
    return result

def setting_rank(eigen_vector):
    minCumSV = 0.9
    current_sum = 0
    sum_list = np.sum(eigen_vector)
    for x in range(len(eigen_vector)):
        current_sum += eigen_vector[x]
        if current_sum > minCumSV * sum_list:
            return x + 1
    return len(eigen_vector)


def applyNorm(X_train, normMat):
    tmp = [np.matmul(X, normMat) for X in X_train]
    return np.asarray(tmp)


def setSeed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def EANorm(X_train, X_test, refer):
    normR = getNormR(refer, refer.shape[-1])
    X_train = applyNorm(X_train, normR)
    X_test = applyNorm(X_test, normR)
    return X_train, X_test


def transformMat(X, Basis, reverse = False):
    tmp = []
    for ii in range(len(X)):
        Xnew = np.copy(X[ii])
        if reverse:
            Xnew = Xnew.T
        U_Test = getV_SVD(Xnew)

        transformMatrix = np.matmul( U_Test, Basis.T)
        Xnew = matmul_list([ Basis.T, transformMatrix, U_Test, Xnew])
        tmp.append(Xnew)
    return np.asarray(tmp)

def normMat(X_train, X_test):
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, mean, std 

def listRepresent(X_train, y_train, reverse = False):
    tmp = []
    for label in np.unique(y_train):
        tmplist = X_train[np.where(y_train == label)]
        meanMat = np.mean( tmplist, axis = 0)
        if reverse:
            meanMat = meanMat.T
        tmp.append(meanMat)
    return np.hstack(tmp)

def getV_SVD(matrix):
    tmpMat = np.matmul(matrix, matrix.T)
    _, Sigma_mean, UmeanMat = np.linalg.svd(tmpMat , full_matrices=False)
    UmeanMat = UmeanMat.T
    return UmeanMat
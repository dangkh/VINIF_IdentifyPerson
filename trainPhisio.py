import numpy as np
import argparse
import os
from distutils.util import strtobool
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from scipy.linalg import sqrtm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import sys
from util.nets import *
import random


from util.dataUtil import *
from util.modelUtil import *
from util.preproc import *


from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

from mne.decoding import CSP

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, SpatialDropout1D, SpatialDropout2D, BatchNormalization
from tensorflow.keras.layers import Flatten, InputSpec, Layer, Concatenate, AveragePooling2D, MaxPooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, LSTM , SeparableConv2D, DepthwiseConv2D, ConvLSTM2D, LayerNormalization
from tensorflow.keras.layers import TimeDistributed, Lambda, AveragePooling1D, GRU, Attention, Dot, Add, Conv1D, Multiply
from tensorflow.keras.constraints import max_norm, unit_norm 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.utils import plot_model

import time
import scipy.io
import scipy
from scipy import stats, fft, signal
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from timeit import default_timer as timer
from EEGITNET import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

channelCombos = [
    ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'T8', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TP7', 'TP8', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'FT7', 'FT8', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'T8', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TP7', 'TP8', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    ['T7', 'T8', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'TP7', 'TP8', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    ['FT7', 'FT8', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'T8', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
    ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
            'Fp1', 'Fpz', 'Fp2', 
            'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 
            'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
            'O1', 'Oz', 'O2', 'Iz']
]

# channelCombos = [
#     ['C3', 'Cz', 'C4', 'CP1', 'CP2'], ['F3', 'F4', 'C3', 'C4'], ['Fp1', 'Fp2', 'F7',
#                                                                  'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8'],
#     ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'Pz', 'Oz', 'O2', 'P8', 'P4', 'CP2', 'CP6', 'T8', 
#     'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'],
#      ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
#             'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
#             'Fp1', 'Fpz', 'Fp2', 
#             'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
#             'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 
#             'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 
#             'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
#             'O1', 'Oz', 'O2', 'Iz']
# ]

listChns = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
            'Fp1', 'Fpz', 'Fp2', 
            'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 
            'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 
            'O1', 'Oz', 'O2', 'Iz']
listMethods = ['PSD + SVM', 'IHAR + SVM']



def extractSub(ids, X_src2, m_src2):
    tmpX = X_src2[m_src2['subject'] == ids+1]
    tmpM = m_src2[m_src2['subject'] == ids+1]
    data = []
    cur = -1
    tmp = []
    listRun = ["run_4", "run_6", "run_8", "run_10", "run_12", "run_14"]
    for ii in range(len(tmpM)):
        scenarioID = ord(tmpM.iloc[ii]['run'][-1]) - ord('0') 
        # scenarioID = listRun.index(tmpM.iloc[ii]['run'])
        if cur != scenarioID and len(tmp) > 0:
            cur = scenarioID
            data.append([scenarioID,tmp, ''])
            tmp = []
        tmp.append(tmpX[ii].T)
    if len(tmp) > 0:
        data.append([ scenarioID, tmp, ''])
    return data


def extractDataPhisio_byInfo(info):
    fmin, fmax = info['bandL'], info['bandR']
    sfreq = 128.
    ds_src2 = PhysionetMI()
    numSub = info['numSub']
    datas = []
    prgm_4classes = MotorImagery(n_classes=4, resample=sfreq, fmin=fmin, fmax=fmax)
    X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=[x+1 for x in range(numSub)])
    for ii in range(numSub):
        tmp = extractSub(ii, X_src2, m_src2)
        datas.append(tmp)
    return datas


def trainCore(X_train, X_test, y_train, y_test, info):
    X_train, X_test, _, _ = normMat(X_train, X_test)
    if args.eaNorm == 'DEA':
        allMat = listRepresent(X_train, y_train, False)
        UmeanMat, _ = getV_SVD(allMat)
        
        X_train = transformMat(X_train, UmeanMat, False)
        X_test = transformMat(X_test, UmeanMat, False)
        X_train, X_test = EANorm(X_train, X_test, X_train)

    elif args.eaNorm == 'EA':
        X_train, X_test = EANorm(X_train, X_test, X_train)

    if args.modelFeatures == 'PSD':
        X_train, y_train, X_test, y_test = PSD(X_train, y_train, X_test, y_test)
    elif args.modelFeatures == 'IHAR':
        X_train, y_train, X_test, y_test = IHAR(X_train, y_train, X_test, y_test, listChns)           
    elif args.modelFeatures == 'APF':
        X_train = np.mean(np.log(np.abs(X_train)), axis = 1)
        X_test = np.mean(np.log(np.abs(X_test)), axis = 1)
    elif args.modelFeatures == 'CSP':
        csp = CSP(n_components=10)
        X_train = csp.fit_transform(X_train, y_train)
        X_test = csp.transform(X_test)

    if args.modelName == 'SVM':
        return SVM(X_train, y_train, X_test, y_test)

    elif args.modelName == 'ITNET':
        X_test = np.transpose(X_test, (0, 2, 1))
        X_train = np.transpose(X_train, (0, 2, 1))
        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
        num_class = len(np.unique(y_train))

        enc = OneHotEncoder()
        y_train = np.asarray(y_train).reshape(-1,1)
        y_test = np.asarray(y_test).reshape(-1,1)
        enc.fit(y_train)
        y_train = enc.transform(y_train).toarray()
        y_test = enc.transform(y_test).toarray()
        _, Chans, Samples, _ = X_train.shape
        #===============================
        # Training folds
        All_model = []
        All_AccuracyTrain = []
        All_AccuracyVal = []
        All_AccuracyTest = []
        All_loss = []
        All_epochs = []
        
        model = Network(Chans, Samples, 'single', num_class)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        mc = ModelCheckpoint('./Results/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        fittedModel = model.fit(X_train, y_train, batch_size = 32, epochs = 10, 
                    verbose = 0)
        probs = model.predict(X_test)
        preds = probs.argmax(axis = -1)  
        return round(100*np.mean(preds == y_test.argmax(axis=-1)),2)
    elif (info['modelName'] == 'CNN' or info['modelName'] == "CNN_LSTM"):
        # print(X_train.shape)
        # X_train = np.expand_dims(X_train, axis=1)
        # X_test = np.expand_dims(X_test, axis=1)
        # print(X_train.shape)
        # stop
        n_samples, n_timestamp, n_channels = X_train.shape
        X_train = X_train.reshape((n_samples, n_timestamp, n_channels, 1))
        X_train = np.transpose(X_train, (0, 3, 1, 2))

        n_samples, n_timestamp, n_channels = X_test.shape
        X_test = X_test.reshape((n_samples, n_timestamp, n_channels, 1))
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        trainLoader, validLoader = TrainTestLoader([X_train, y_train, X_test, y_test])
        num_class = len(np.unique(y_train))

        listModelName = []
        model = chooseModel(str(args.modelName), num_class=num_class, input_size=(1, X_train.shape[2], X_train.shape[3]))
        print("Model architecture >>>", model)
        stopppp
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 3e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 10
        _, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader,
                                            validLoader, n_class=num_class, log_batch=max(len(trainLoader) // 30, 1))
        return acc
    elif info['modelName'] == 'SHALLOW':
        n_samples, n_timestamp, n_channels = X_train.shape
        X_train = X_train.reshape((n_samples, n_timestamp, n_channels, 1))
        X_train = np.transpose(X_train, (0, 2, 1, 3))
        n_samples, n_timestamp, n_channels = X_test.shape
        X_test = X_test.reshape((n_samples, n_timestamp, n_channels, 1))
        X_test = np.transpose(X_test, (0, 2, 1, 3))
        trainLoader, validLoader = TrainTestLoader([X_train, y_train, X_test, y_test])
        num_class = len(np.unique(y_train))
        input_time_length = X_train.shape[2]
        in_chans = X_train.shape[3]
        model = EEGShallowClassifier(in_chans=32, n_classes=num_class, input_time_length=128, return_feature=False)
        print("Model architecture >>>", model)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 3e-3
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5 * 0.001)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 20

        _, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader,
                                            validLoader, n_class=num_class, log_batch=len(trainLoader) // 30)
        return acc
    else:
        pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--modelName', help='name of model : {}'.format(listMethods))
    parser.add_argument('--modelFeatures', help='name of features : PSD, IHAR, APF, RAW', default='RAW')
    parser.add_argument('--numSub', help='number of Subject', default=5, type=int)
    parser.add_argument('--numChan', help='number of channel', default=-1, type=int)
    parser.add_argument('--bandL', help='band filter', default=4.0, type=float)
    parser.add_argument('--bandR', help='band filter', default=50.0, type=float)
    parser.add_argument('--eaNorm', help='EA norm', default='False')
    parser.add_argument('--channelType', help='channel seclection in : {}'.format(channelCombos), default=-1, type=int)
    parser.add_argument('--windowSize', help='windowSize', default=128, type=int)
    parser.add_argument('--windowIHAR', help='windowIHAR', default=10, type=int)
    parser.add_argument('--thinking', help='thinking: True. resting: False', default='False')
    parser.add_argument('--trainTestSeperate', help='train first then test. if not, train and test are splitted randomly', default='False')
    parser.add_argument('--output', help='train test are splitted by session', default='./result.txt')
    args = parser.parse_args()
    print(args)
    typeTest = 'trainTestRandom'
    if strtobool(args.trainTestSeperate):
        typeTest = 'trainTestSeperate'

    dataName = f'./PHISYO__numberSub{str(args.numSub)}_band_{str(args.bandL)}_{str(args.bandR)}_channelType_{str(args.channelType)}_{typeTest}'
    if strtobool(args.thinking):
        dataName += '_thinking'
    dataRaw = dataName +'_RAW.npy'
    # duration
    # if int(args.windowSize) != 128:
    #     dataName += f'_size{args.windowSize}'
    # normal
    dataLink = dataName + '.npy'
    print(dataLink)
    info = {
            'bandL': args.bandL,
            'bandR': args.bandR,
            'windowSize': args.windowSize,
            'EA': str(args.eaNorm),
            'channelType': channelCombos[args.channelType],
            'modelName': args.modelName, 
            'typeTest': typeTest,
            'dataset':'Phi',
            'numSub': args.numSub, 
            'numChan': args.numChan, 
            'thinking': strtobool(args.thinking)
        }
    if not os.path.exists(dataLink):
        # normal
        datas = extractDataPhisio_byInfo(info)
        # vary window size
        # if not os.path.exists(dataRaw):            
        #     datas = extractDataPhisio_byInfo(info)
        #     np.save(dataRaw, datas)
        # else:
        #     datas = np.load(dataRaw, allow_pickle=True)    
        print("Number of subjects in data: ", len(datas))
        PreProDatas = preprocessDataInfo(datas, info)
        np.save(dataLink, PreProDatas)
    else:
        PreProDatas = np.load(dataLink, allow_pickle=True)

    listAcc = []
    listSeed = [x*500+15 for x in range(50)]
    numTest = 10
    if typeTest == 'trainTestSeperate':
        numTest = 1
    for testingTime in range(numTest):
        if typeTest == 'trainTestRandom':
            print("Training at {} round".format(testingTime))
            X_f, y_f = getData_All(PreProDatas)
            if int(args.numChan) != -1:
                X_f = X_f[:, :, :int(args.numChan)]
            else:
                listChan2Index = [listChns.index(x) for x in channelCombos[args.channelType]]
                X_f = X_f[:, :, listChan2Index]
            X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.2, random_state= listSeed[testingTime])            
            acc = trainCore(X_train, X_test, y_train, y_test, info)
            print(acc)
            listAcc.append(acc)
        elif typeTest == 'trainTestSeperate':
            print("Training at {} round".format(testingTime))
            for scenario in range(6):
                print("Validate on 5 scenario")
                X_train, y_train, X_test, y_test = getDataScenario(PreProDatas, scenario)
                if int(args.numChan) != -1:
                    X_train = X_train[:, :, :int(args.numChan)]
                    X_test = X_test[:, :, :int(args.numChan)]
                else:
                    listChan2Index = [listChns.index(x) for x in channelCombos[args.channelType]]
                    X_train = X_train[:, :, listChan2Index]
                    X_test = X_test[:, :, listChan2Index]
                print(X_train.shape, X_test.shape)
                acc = trainCore(X_train, X_test, y_train, y_test, info)
                print("Scenario {} with acc: {}".format(scenario, acc))
                listAcc.append(acc)
        # else:
        #     print("Training at {} round".format(testingTime))
        #     setSeed(listSeed[testingTime])
        #     X_train, y_train, X_test, y_test = getDataFuture(PreProDatas, info)
        #     stop
        #     # X_train, y_train = augmentData(X_train, y_train, [3])
        #     # analyzeTrainData(y_train)
        #     acc = trainCore(X_train, X_test, y_train, y_test, info)
        #     print(" acc: {}".format( acc))
        #     listAcc.append(acc)

    listAcc = np.asarray(listAcc)
    sourceFile = open(args.output, 'a')
    print('*'*10, 'Result' ,'*'*10, file = sourceFile)
    print(args, file = sourceFile)
    print(listAcc, file = sourceFile)
    print("Result: ", np.mean(listAcc), np.max(listAcc) - np.mean(listAcc), file = sourceFile)
    print('*'*10, 'End' ,'*'*10, file = sourceFile)
    sourceFile.close()

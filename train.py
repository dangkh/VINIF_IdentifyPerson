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


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

channelCombos = [
    ['Fz', 'Fp1', 'F7', 'F3', 'FC1', 'FC5', 'FT9', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2'],
    ['P3', 'P7', 'PO9', 'Pz', 'PO10', 'P8', 'P4'],
    ['Cz', 'C3', 'CP5', 'CP1', 'CP2', 'CP6', 'C4'],
    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
     'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'],
     ['T7', 'T8']
]

listChns = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
     'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

listMethods = ['PSD + SVM', 'IHAR + SVM']


def trainCore(X_train, X_test, y_train, y_test, info):
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    if args.eaNorm == 'DEA':
        dataLink = dataName + '_COV_DEA.txt'
        tmp = []
        for label in np.unique(y_train):
            tmplist = X_train[np.where(y_train == label)]
            np.random.shuffle(tmplist)
            tmp.append(np.mean( tmplist, axis = 0).T)
        
        allMat = np.hstack(tmp)
        tmpMat = np.matmul(allMat, allMat.T)
        _, Sigma_mean, UmeanMat = np.linalg.svd(tmpMat , full_matrices=False)
        UmeanMat = UmeanMat.T

        tmp = []
        for ii in range(len(X_train)):
            Xnew = np.copy(X_train[ii].T)
            tmpMat = np.matmul(Xnew, Xnew.T)

            _, Sigma_Test, U_Test = np.linalg.svd(tmpMat , full_matrices=False)
            U_Test = U_Test.T
            transformMatrix = np.matmul( U_Test, UmeanMat.T)
            Xnew = matmul_list([ UmeanMat.T, transformMatrix, U_Test, Xnew])
            tmp.append(Xnew.T)
        
        X_train = np.asarray(tmp)

        tmp = []
        for ii in range(len(X_test)):
            Xnew = np.copy(X_test[ii].T)
            tmpMat = np.matmul(Xnew, Xnew.T)

            _, Sigma_Test, U_Test = np.linalg.svd(tmpMat , full_matrices=False)
            U_Test = U_Test.T
            transformMatrix = np.matmul( U_Test, UmeanMat.T)
            Xnew = matmul_list([ UmeanMat.T, transformMatrix, U_Test, Xnew])
            tmp.append(Xnew.T)
        X_test = np.asarray(tmp)
        normR = getNormR(X_train, X_train.shape[-1])

        X_train = applyNorm(X_train, normR)
        X_test = applyNorm(X_test, normR)

    elif args.eaNorm == 'EA':
        dataLink = dataName + '_COV_EA.txt'
        normR = getNormR(X_train, X_train.shape[-1])
        tmp = []
        for ii in range(len(X_train)):
            Xnew = np.matmul(X_train[ii], normR)
            tmp.append(Xnew)
        X_train = np.asarray(tmp)
        tmp = []
        for ii in range(len(X_test)):
            Xnew = np.matmul(X_test[ii], normR)
            tmp.append(Xnew)
        X_test = np.asarray(tmp)

    if args.modelName == 'PSD':
        return PSD(X_train, y_train, X_test, y_test)

        
    elif args.modelName == 'IHAR':
        return IHAR(X_train, y_train, X_test, y_test, listChns)
    elif args.modelName == 'WLD':
        return -1
        pass

    elif (info['modelName'] == 'CNN' or info['modelName'] == "CNN_LSTM"):
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
        # X_train, y_train = augmentData(X_train, y_train, np.unique(y_train))
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
    parser.add_argument('--input', help='input data dir')
    parser.add_argument('--modelName', help='name of model : {}'.format(listMethods))
    parser.add_argument('--bandL', help='band filter', default=4.0, type=float)
    parser.add_argument('--bandR', help='band filter', default=50.0, type=float)
    parser.add_argument('--eaNorm', help='EA norm', default='False')
    parser.add_argument('--channelType', help='channel seclection in : {}'.format(channelCombos), default=3, type=int)
    parser.add_argument('--windowSize', help='windowSize', default=128, type=int)
    parser.add_argument('--windowIHAR', help='windowIHAR', default=10, type=int)
    parser.add_argument('--extractFixation', help='type of extraction in eeg. Fixation: True. All: False', default='False')
    parser.add_argument('--thinking', help='thinking: True. resting: False', default='False')
    parser.add_argument('--trainTestSeperate', help='train first then test. if not, train and test are splitted randomly', default='False')
    parser.add_argument('--trainTestSession', help='train test are splitted by session', default='True')
    parser.add_argument('--output', help='train test are splitted by session', default='./result.txt')
    args = parser.parse_args()
    print(args)
    '''
    # python train.py --windowSize 128 --modelName PSD --bandL 0.1 --bandR 50 --extractFixation False --thinking False --trainTestSeperate False --trainTestSession False
    '''
    listPaths = []
    numberObject = 50
    counter = 0

    prePath = args.input
    for x in os.listdir(prePath):
        if x != "BN001" and x != "K317" and x != "BN002" and x != "K299" and x != "K305":
            listPaths.append(prePath + '/' + x)
            counter += 1
            if counter > numberObject:
                break

    listPaths = sorted(listPaths)
    tmpExtract = 'Fixation'
    if not strtobool(args.extractFixation):
        tmpExtract = 'All'
    typeTest = 'trainTestRandom'
    if strtobool(args.trainTestSeperate):
        typeTest = 'trainTestSeperate'
    if strtobool(args.trainTestSession) and strtobool(args.trainTestSeperate):
        typeTest = 'trainTestSession'

    dataName = './' + 'band_' + str(args.bandL) + '_' + str(args.bandR) + '_channelType_' + str(args.channelType) + '_' + typeTest + '_' + tmpExtract
    if strtobool(args.thinking):
        dataName += '_thinking'
    dataRaw = dataName +'_RAW.npy'
    # duration
    # if int(args.windowSize) != 128:
    # dataName += f'_size{args.windowSize}'
    # normal
    dataLink = dataName + '.npy'
    # test
    # dataLink = 'test.npy'
    print(dataLink)
    info = {
            'bandL': args.bandL,
            'bandR': args.bandR,
            'windowSize': args.windowSize,
            'listPaths': listPaths,
            'EA': str(args.eaNorm),
            'extractFixation': strtobool(args.extractFixation),
            'channelType': channelCombos[args.channelType],
            'modelName': args.modelName, 
            'typeTest': typeTest,
            'thinking': strtobool(args.thinking)
        }
    if not os.path.exists(dataLink):
        # normal
        datas = extractData_byInfo(info)

        # vary window size
        # if not os.path.exists(dataRaw):            
        #     datas = extractData_byInfo(info)
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

    for testingTime in range(numTest):
        if typeTest == 'trainTestRandom':
            print("Training at {} round".format(testingTime))
            X_f, y_f = getData_All(PreProDatas)
            X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.2, random_state= listSeed[testingTime])            
            acc = trainCore(X_train, X_test, y_train, y_test, info)
            print(acc)
            listAcc.append(acc)
        elif typeTest == 'trainTestSeperate':
            print("Training at {} round".format(testingTime))
            for scenario in range(9):
                X_train, y_train, X_test, y_test = getDataScenario(PreProDatas, scenario)
                print(X_train.shape)
                acc = trainCore(X_train, X_test, y_train, y_test, info)
                print("Scenario {} with acc: {}".format(scenario, acc))
                listAcc.append(acc)
            break
        # else:
        #     print("Training at {} round".format(testingTime))
        #     setSeed(listSeed[testingTime])
        #     X_train, y_train, X_test, y_test = getDataFuture(PreProDatas, info)
        #     acc = trainCore(X_train, X_test, y_train, y_test, info)
        #     print(" acc: {}".format( acc))
        #     listAcc.append(acc)

    listAcc = np.asarray(listAcc)
    sourceFile = open(args.output, 'a')
    print('*'*10, 'Result' ,'*'*10, file = sourceFile)
    print(args, file = sourceFile)
    print(listAcc, file = sourceFile)
    print(np.mean(listAcc), np.max(listAcc) - np.mean(listAcc), file = sourceFile)
    print('*'*10, 'End' ,'*'*10, file = sourceFile)
    sourceFile.close()

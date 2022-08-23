import numpy as np
import argparse
import os
from ultis import *
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
from ultis import *
from nets import *
import random


from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

channelCombos = [
    ['C3', 'Cz', 'C4', 'CP1', 'CP2'], ['F3', 'F4', 'C3', 'C4'], ['Fp1', 'Fp2', 'F7',
                                                                 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8'],
    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
     'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
]

listMethods = ['PSD + SVM', 'IHAR + SVM']


def SVM(X_train, y_train, X_test, y_test):
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted)
    counter = 0
    for i in range(len(predicted)):
        if predicted[i] == y_test[i]:
            counter += 1
    acc = counter * 100.0 / len(predicted)
    return acc


def PSD(X_train, y_train, X_test, y_test):
    tmp = []
    for xxx in X_train:
        xx = xxx.T
        fft_rs, freq = GetFFT(xx)
        newX = GetPSD(fft_rs)
        newX = np.asarray(newX)
        newX = newX.real
        newX = newX.reshape(-1)
        tmp.append(newX)
    X_train = np.vstack(tmp)

    tmp = []
    for xxx in X_test:
        xx = xxx.T
        fft_rs, freq = GetFFT(xx)
        newX = GetPSD(fft_rs)
        newX = np.asarray(newX)
        newX = newX.real
        newX = newX.reshape(-1)
        tmp.append(newX)
    X_test = np.vstack(tmp)
    return SVM(X_train, y_train, X_test, y_test)


def IHAR(X_train, y_train, X_test, y_test):
    electrodeIHAR = [['Fp1', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1'], ['Fp2', 'F8', 'F4', 'FC6', 'T8', 'P8', 'O2']]
    numNode = len(electrodeIHAR[0])
    electrodeIndex = []
    for electrodes in electrodeIHAR:
        electrodeIndex.append([channelCombos[-1].index(x) for x in electrodes])
    tmp = []
    for xxx in X_train:
        xx = xxx.T
        fft_rs, freq = GetFFT(xx)
        newX = GetPSD(fft_rs)
        newX = np.asarray(newX)
        newX = newX.real
        newX = MA(newX, args.windowIHAR)
        IharTrain = []
        for ii in range(numNode):
            left = newX[electrodeIndex[0][ii]]
            right = newX[electrodeIndex[1][ii]]
            ihar = left / right
            IharTrain.append(ihar)
        # IharTrain.append(newX)
        newX = np.vstack([np.vstack(IharTrain), newX])
        newX = newX.reshape(-1)
        tmp.append(newX)
    X_train = np.vstack(tmp)

    tmp = []
    for xxx in X_test:
        xx = xxx.T
        fft_rs, freq = GetFFT(xx)
        newX = GetPSD(fft_rs)
        newX = np.asarray(newX)
        newX = newX.real
        newX = MA(newX, args.windowIHAR)
        IharTrain = []
        for ii in range(numNode):
            left = newX[electrodeIndex[0][ii]]
            right = newX[electrodeIndex[1][ii]]
            ihar = left / right
            IharTrain.append(ihar)
        # IharTrain.append(newX)
        newX = np.vstack([np.vstack(IharTrain), newX])
        newX = newX.reshape(-1)
        tmp.append(newX)
    X_test = np.vstack(tmp)    
    return SVM(X_train, y_train, X_test, y_test)


def applyNorm(X_train, normMat):
    tmp = []
    for ii in range(len(X_train)):
        Xnew = np.matmul(X_train[ii], normMat)
        tmp.append(Xnew)
    return np.asarray(tmp)

def setSeed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def extractSub(ids, X_src2, m_src2):
    tmpX = X_src2[m_src2['subject'] == ids+1]
    tmpM = m_src2[m_src2['subject'] == ids+1]
    data = []
    cur = -1
    tmp = []
    for ii in range(len(tmpM)):
        scenarioID = ord(tmpM.iloc[ii]['run'][-1]) - ord('0') 
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
    numSub = 50
    datas = []
    prgm_4classes = MotorImagery(n_classes=4, resample=sfreq, fmin=fmin, fmax=fmax)
    X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=[x+1 for x in range(numSub)])
    for ii in range(numSub):
        tmp = extractSub(ii, X_src2, m_src2)
        datas.append(tmp)
    return datas


def trainCore(X_train, X_test, y_train, y_test, info):
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    if args.eaNorm == 'DEA':
        dataLink = dataName + '_COV_DEA.txt'
        # normR = getNormR(X_train)

        # X_train = applyNorm(X_train, normR)
        # X_test = applyNorm(X_test, normR)
        tmp = []
        for label in np.unique(y_train):
            tmplist = X_train[np.where(y_train == label)]
            np.random.shuffle(tmplist)
            tmp.append(np.mean( tmplist, axis = 0))
        
        # v1

        allMat = np.hstack(tmp)
        tmpMat = np.matmul(allMat, allMat.T)
        _, Sigma_mean, UmeanMat = np.linalg.svd(tmpMat , full_matrices=False)
        UmeanMat = UmeanMat.T
        # print(np.sum(np.abs(tmpMat - matmul_list([UmeanMat, Sigma_mean * Sigma_mean, UmeanMat.T]))))
        # print(Sigma_mean)
        # stop
        numCov = setting_rank(Sigma_mean)
        # # numCov = 100
        # UmeanMat = UmeanMat[:numCov]

        tmp = []
        for ii in range(len(X_train)):
            Xnew = np.copy(X_train[ii])
            tmpMat = np.matmul(Xnew, Xnew.T)

            _, Sigma_Test, U_Test = np.linalg.svd(tmpMat , full_matrices=False)
            U_Test = U_Test.T
            # U_Test = U_Test[:numCov]
            transformMatrix = np.matmul( U_Test, UmeanMat.T)
            Xnew = matmul_list([ UmeanMat.T, transformMatrix, U_Test, Xnew])
            tmp.append(Xnew)
        
        X_train = np.asarray(tmp)
        # normR = getNormR(X_train)
        # X_train = applyNorm(X_train, normR)

        tmp = []
        for ii in range(len(X_test)):
            Xnew = np.copy(X_test[ii])
            tmpMat = np.matmul(Xnew, Xnew.T)

            _, Sigma_Test, U_Test = np.linalg.svd(tmpMat , full_matrices=False)
            U_Test = U_Test.T
            # U_Test = U_Test[:numCov]
            transformMatrix = np.matmul( U_Test, UmeanMat.T)
            Xnew = matmul_list([ UmeanMat.T, transformMatrix, U_Test, Xnew])
            tmp.append(Xnew)
        X_test = np.asarray(tmp)
        # X_test = applyNorm(X_test, normR)

        normR = getNormR(X_train, 64)

        X_train = applyNorm(X_train, normR)
        X_test = applyNorm(X_test, normR)

    elif args.eaNorm == 'EA':
        dataLink = dataName + '_COV_EA.txt'
        normR = getNormR(X_train, 64)
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
        # model PSD + SVM
        return PSD(X_train, y_train, X_test, y_test)

        
    elif args.modelName == 'IHAR':
        # model IHAR + SVM
        return IHAR(X_train, y_train, X_test, y_test)           
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
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 20
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
    parser.add_argument('--modelName', help='name of model : {}'.format(listMethods))
    parser.add_argument('--bandL', help='band filter', default=4.0, type=float)
    parser.add_argument('--bandR', help='band filter', default=50.0, type=float)
    parser.add_argument('--eaNorm', help='EA norm', default='False')
    parser.add_argument('--channelType', help='channel seclection in : {}'.format(channelCombos), default=3, type=int)
    parser.add_argument('--windowSize', help='windowSize', default=128, type=int)
    parser.add_argument('--windowIHAR', help='windowIHAR', default=10, type=int)
    parser.add_argument('--thinking', help='thinking: True. resting: False', default='False')
    parser.add_argument('--trainTestSeperate', help='train first then test. if not, train and test are splitted randomly', default='False')
    parser.add_argument('--output', help='train test are splitted by session', default='./result.txt')
    args = parser.parse_args()
    print(args)
    '''
    # python trainPhisio.py  --modelName IHAR --output ./res/restingHMI.txt --bandL 13 --bandR 30 --thinking False --trainTestSeperate False --windowSize 128  --eaNorm DEA
    '''
    typeTest = 'trainTestRandom'
    if strtobool(args.trainTestSeperate):
        typeTest = 'trainTestSeperate'

    dataName = f'./PHISYO_band_{str(args.bandL)}_{str(args.bandR)}_channelType_{str(args.channelType)}_{typeTest}'
    if strtobool(args.thinking):
        dataName += '_thinking'
    dataRaw = dataName +'_RAW.npy'
    # duration
    # if int(args.windowSize) != 128:
    # dataName += f'_size{args.windowSize}'
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
            'thinking': strtobool(args.thinking)
        }
    if not os.path.exists(dataLink):
        # normal
        datas = extractDataPhisio_byInfo(info)
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
        else:
            print("Training at {} round".format(testingTime))
            setSeed(listSeed[testingTime])
            X_train, y_train, X_test, y_test = getDataFuture(PreProDatas, info)
            stop
            # X_train, y_train = augmentData(X_train, y_train, [3])
            # analyzeTrainData(y_train)
            acc = trainCore(X_train, X_test, y_train, y_test, info)
            print(" acc: {}".format( acc))
            listAcc.append(acc)

    listAcc = np.asarray(listAcc)
    sourceFile = open(args.output, 'a')
    print('*'*10, 'Result' ,'*'*10, file = sourceFile)
    print(args, file = sourceFile)
    print(listAcc, file = sourceFile)
    print(np.mean(listAcc), np.max(listAcc) - np.mean(listAcc), file = sourceFile)
    print('*'*10, 'End' ,'*'*10, file = sourceFile)
    sourceFile.close()
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

channelCombos = [
    ['C3', 'Cz', 'C4', 'CP1', 'CP2'], ['F3', 'F4', 'C3', 'C4'], ['Fp1', 'Fp2', 'F7',
                                                                 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8'],
    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
     'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
]

listMethods = ['PSD + SVM', 'IHAR + SVM']
persons = [10, 9, 6]


def setSeed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--input', help='input data dir')
    parser.add_argument('--modelName', help='name of model : {}'.format(listMethods))
    parser.add_argument('--bandL', help='band filter', default=4.0, type=float)
    parser.add_argument('--bandR', help='band filter', default=50.0, type=float)
    parser.add_argument('--eaNorm', help='EA norm', default='False')
    parser.add_argument('--channelType', help='channel seclection in : {}'.format(channelCombos), default=3, type=int)
    parser.add_argument('--windowSize', help='windowSize', default=120, type=int)
    parser.add_argument('--windowIHAR', help='windowIHAR', default=10, type=int)
    parser.add_argument('--extractFixation', help='type of extraction in eeg. Fixation: True. All: False', default='False')
    parser.add_argument('--trainTestSeperate', help='train first then test. if not, train and test are splitted randomly', default='False')
    parser.add_argument('--trainTestSession', help='train test are splitted by session', default='True')
    args = parser.parse_args()
    print(args)

    listPaths = []
    numberObject = 20
    counter = 0

    prePath = args.input
    for x in os.listdir(prePath):
        if x != "BN001" and x != "K317" and x != "BN002" and x != "K299" and x != "K305":
            listPaths.append(prePath + '/' + x)
            counter += 1
            if counter > numberObject:
                break

    tmpExtract = 'Fixation'
    if not strtobool(args.extractFixation):
        tmpExtract = 'All'
    tmp = 'trainTestRandom'
    if strtobool(args.trainTestSeperate):
        tmp = 'trainTestSeperate'
    if strtobool(args.trainTestSession) and strtobool(args.trainTestSeperate):
        tmp = 'trainTestSession'

    dataName = './' + 'band_' + str(args.bandL) + '_' + str(args.bandR) + '_channelType_' + str(args.channelType) + '_' + tmp + '_' + tmpExtract
    dataLink = dataName + '.npy'
    print(dataLink)
    if not os.path.exists(dataLink):
        info = {
            'bandL': args.bandL,
            'bandR': args.bandR,
            'windowSize': args.windowSize,
            'listPaths': listPaths,
            'EA': strtobool(args.eaNorm),
            'extractFixation': strtobool(args.extractFixation),
            'channelType': channelCombos[args.channelType]
        }
        datas = extractData_byInfo(info)
        print("Number of subjects in data: ", len(datas))
        PreProDatas = preprocessDataInfo(datas, info)
        np.save(dataLink, PreProDatas)
    else:
        PreProDatas = np.load(dataLink, allow_pickle=True)

    if strtobool(args.trainTestSeperate):
        X_f, y_f = getData_All(PreProDatas)
    else:
        X_f, y_f = getData_All(PreProDatas)
        X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.2, random_state=50)

    # mean = np.mean(np.mean(X_train, axis = 2, keepdims = False), axis = 0)
    mean = np.mean(X_train, axis=0, keepdims=True)
    # mean = mean.reshape(len(mean), 1)
    # meanMat = np.matmul(mean, np.ones([1, 32]))
    # std = np.mean(np.std(X_train, axis = 2, keepdims = False), axis = 0)
    std = np.std(X_train, axis=0, keepdims=True)
    # std = std.reshape(len(std), 1)
    # stdMat = np.matmul(std, np.ones([1, 32]))
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    meanMat = np.mean(X_train, axis=0, keepdims=False)
    _, Sigma_mean, UmeanMat = np.linalg.svd(meanMat.T / np.sqrt(meanMat.T.shape[0] - 1), full_matrices=False)
    UmeanMat = UmeanMat.T
    ksmall = 50

    if strtobool(args.eaNorm):
        dataLink = dataName + '_COV.txt'
        normR = getNormR(X_train)
        normR = sqrtm(normR)
        normR = np.linalg.inv(normR)
        # if not os.path.exists(dataLink):
        #     normR = getNormR(X_train)
        #     normR = sqrtm(normR)
        #     normR = np.linalg.inv(normR)
        #     np.savetxt(dataLink, normR, fmt= '%.5f')
        # else:
        #     normR = np.loadtxt(dataLink)

        tmp = []
        for ii in range(len(X_train)):
            # Xnew = np.matmul(X_train[ii], normR)
            # tmp.append(Xnew)
            Xnew = np.copy(X_train[ii])
            _, Sigma_Test, U_Test = np.linalg.svd(Xnew.T / np.sqrt(Xnew.T.shape[0] - 1), full_matrices=False)
            U_Test = U_Test.T
            U_Test = U_Test[:, :ksmall]
            UmeanTmp = UmeanMat[:, :ksmall]
            transformMatrix = np.matmul(U_Test.T, UmeanTmp)
            Xnew = matmul_list([Xnew.T, UmeanTmp, transformMatrix, U_Test.T])
            Xnew = Xnew.T
            tmp.append(Xnew)
        X_train = np.asarray(tmp)
        normR = getNormR(X_train)
        normR = sqrtm(normR)
        normR = np.linalg.inv(normR)
        tmp = []
        for ii in range(len(X_train)):
            Xnew = np.matmul(X_train[ii], normR)
            tmp.append(Xnew)
        tmp = []
        for ii in range(len(X_test)):
            # Xnew = np.matmul(X_test[ii], normR)
            # tmp.append(Xnew)
            Xnew = np.copy(X_test[ii])
            _, Sigma_Test, U_Test = np.linalg.svd(Xnew.T / np.sqrt(Xnew.T.shape[0] - 1), full_matrices=False)
            U_Test = U_Test.T
            U_Test = U_Test[:, :ksmall]
            UmeanTmp = UmeanMat[:, :ksmall]
            transformMatrix = np.matmul(U_Test.T, UmeanTmp)
            Xnew = matmul_list([Xnew.T, UmeanTmp, transformMatrix, U_Test.T])
            Xnew = Xnew.T
            Xnew = np.matmul(Xnew, normR)
            tmp.append(Xnew)
        X_test = np.asarray(tmp)

    if args.modelName == 'PSD':
        # model PSD + SVM
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

        clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        predicted = np.asarray(predicted)
        counter = 0
        for i in range(len(predicted)):
            if predicted[i] == y_test[i]:
                counter += 1
        print(counter * 100.0 / len(predicted))
    elif args.modelName == 'IHAR':
        # model IHAR + SVM
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

        clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        predicted = np.asarray(predicted)
        counter = 0
        for i in range(len(predicted)):
            if predicted[i] == y_test[i]:
                counter += 1
        print(counter * 100.0 / len(predicted))
    elif args.modelName == 'WLD':
        import pywt
        tmp = []
        for ii in range(len(X_train)):
            coeffs = pywt.wavedec2(X_train[ii].T, 'db4')
            tmpCoe = []
            for level in range(1, 2):
                if level == 0:
                    cA = coeffs[level]
                    tmpCoe.append(cA.reshape(-1))
                else:
                    (cH, cV, cD) = coeffs[level]
                    tmpCoe.append(cH.reshape(-1))
                    tmpCoe.append(cV.reshape(-1))
                    tmpCoe.append(cD.reshape(-1))
            Xnew = np.log(np.hstack(tmpCoe))
            tmp.append(Xnew)
        X_train = np.vstack(tmp)

        tmp = []
        for ii in range(len(X_test)):
            coeffs = pywt.wavedec2(X_test[ii].T, 'db4')
            tmpCoe = []
            for level in range(1, 2):
                if level == 0:
                    cA = coeffs[level]
                    tmpCoe.append(cA.reshape(-1))
                else:
                    (cH, cV, cD) = coeffs[level]
                    tmpCoe.append(cH.reshape(-1))
                    tmpCoe.append(cV.reshape(-1))
                    tmpCoe.append(cD.reshape(-1))
            Xnew = np.log(np.hstack(tmpCoe))
            tmp.append(Xnew)
        X_test = np.vstack(tmp)

        mean = np.mean(np.mean(X_train, axis=2, keepdims=False), axis=0)
        # mean = np.mean(X_train, axis = 0, keepdims = True)
        mean = mean.reshape(len(mean), 1)
        meanMat = np.matmul(mean, np.ones([1, 32]))
        std = np.mean(np.std(X_train, axis=2, keepdims=False), axis=0)
        # std = np.std(X_train, axis = 0, keepdims = True)
        std = std.reshape(len(std), 1)
        stdMat = np.matmul(std, np.ones([1, 32]))
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        predicted = np.asarray(predicted)
        counter = 0
        for i in range(len(predicted)):
            if predicted[i] == y_test[i]:
                counter += 1
        print(counter * 100.0 / len(predicted))

    elif args.modelName == 'CNN':
        setSeed(10000)
        n_samples, n_timestamp, n_channels = X_train.shape
        X_train = X_train.reshape((n_samples, n_timestamp, n_channels, 1))
        X_train = np.transpose(X_train, (0, 3, 1, 2))

        n_samples, n_timestamp, n_channels = X_test.shape
        X_test = X_test.reshape((n_samples, n_timestamp, n_channels, 1))
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        trainLoader, validLoader = TrainTestLoader([X_train, y_train, X_test, y_test])
        num_class = len(np.unique(y_train))

        listModelName = []
        model = chooseModel("CNN_LSTM", num_class=num_class, input_size=(1, X_train.shape[2], X_train.shape[3]))
        print("Model architecture >>>", model)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 3e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 20

        _, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader,
                                            validLoader, n_class=num_class, log_batch=len(trainLoader) // 30)
        print(acc)
        print(accTrain)
    elif args.modelName == 'SHALLOW':
        setSeed(10000)
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
        lr = 3e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5 * 0.001)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 20

        _, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader,
                                            validLoader, n_class=num_class, log_batch=len(trainLoader) // 30)
        print(acc)
        print(accTrain)
    else:
        pass

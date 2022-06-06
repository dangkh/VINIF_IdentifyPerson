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

channelCombos = [   
                    ['C3', 'Cz', 'C4', 'CP1', 'CP2'], ['F3', 'F4', 'C3', 'C4'], ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8'], 
                    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 
                    'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
                ]

listMethods = ['PSD + SVM', 'IHAR + SVM']
persons = [10, 9, 6]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'train')
    parser.add_argument('--input', help = 'input data dir')
    parser.add_argument('--modelName', help = 'name of model : {}'.format(listMethods))
    parser.add_argument('--bandL', help = 'band filter', default = 4.0, type= float)
    parser.add_argument('--bandR', help = 'band filter', default = 50.0, type= float)
    parser.add_argument('--eaNorm', help = 'EA norm', default = 'False')
    parser.add_argument('--channelType', help = 'channel seclection in : {}'.format(channelCombos), default = 3, type=int)
    parser.add_argument('--windowSize', help = 'windowSize', default = 120, type=int)
    parser.add_argument('--windowIHAR', help = 'windowIHAR', default = 10, type=int)
    parser.add_argument('--extractFixation', help = 'type of extraction in eeg. Fixation: True. All: False', default = 'False')
    parser.add_argument('--trainTestSeperate', help = 'train first then test. if not, train and test are splitted randomly', default = 'False')
    parser.add_argument('--trainTestSession', help = 'train test are splitted by session', default = 'True')
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
            'bandL':        args.bandL,
            'bandR':        args.bandR, 
            'windowSize':   args.windowSize,
            'listPaths':    listPaths,
            'EA':           strtobool(args.eaNorm),
            'extractFixation':  strtobool(args.extractFixation),
            'channelType':  channelCombos[args.channelType]
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
        X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.2, random_state=42)

    mean = np.mean(X_train, axis = 0, keepdims = True)
    std = np.std(X_train, axis = 0, keepdims = True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    if strtobool(args.eaNorm):
        dataLink = dataName + '_COV.txt'   
        if not os.path.exists(dataLink):
            normR = getNormR(X_train)
            normR = sqrtm(normR)
            normR = np.linalg.inv(normR)
            np.savetxt(dataLink, normR, fmt= '%.5f')
        else:
            normR = np.loadtxt(dataLink)
            print(normR)

        tmp = []
        for ii in range(len(X_train)):
            Xnew = np.matmul(normR, X_train[ii])
            tmp.append(Xnew)
        X_train = np.asarray(tmp)
        tmp = []
        for ii in range(len(X_test)):
            Xnew = np.matmul(normR, X_test[ii])
            tmp.append(Xnew)
        X_test = np.asarray(tmp)

    
    if args.modelName == '1':
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
    else:
        # model IHAR + SVM
        electrodeIHAR = [[ 'Fp1', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1'], [ 'Fp2', 'F8', 'F4', 'FC6', 'T8', 'P8', 'O2']]
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
            newX = MA(newX, args.windowIHAR)
            for ii in range(numNode):
                left = 
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
            newX = moving_average(newX, args.windowIHAR)
            newX = newX.real
            newX = newX.reshape(-1)
            tmp.append(newX)
        X_test = np.vstack(tmp)
    # 
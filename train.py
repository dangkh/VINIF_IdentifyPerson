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

channelCombos = [   
                    ['C3', 'Cz', 'C4', 'CP1', 'CP2'], ['F3', 'F4', 'C3', 'C4'], ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8'], 
                    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 
                    'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
                ]
persons = [10, 9, 6]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'train')
    parser.add_argument('--input', help = 'input data dir')
    parser.add_argument('--modelName', help = 'name of model')
    parser.add_argument('--bandL', help = 'band filter', default = 4.0, type= float)
    parser.add_argument('--bandR', help = 'band filter', default = 50.0, type= float)
    parser.add_argument('--eaNorm', help = 'EA norm', default = 'False')
    parser.add_argument('--channelType', help = 'channel seclection in : {}'.format(channelCombos), default = 3, type=int)
    parser.add_argument('--windowSize', help = 'windowSize', default = 120, type=int)
    parser.add_argument('--extractFixation', help = 'type of extraction in eeg. Fixation: True. All: False', default = 'False')
    parser.add_argument('--trainTestSeperate', help = 'train first then test. if not, train and test are splitted randomly', default = 'False')
    parser.add_argument('--trainTestSession', help = 'train test are splitted by session', default = 'True')
    args = parser.parse_args()
    print(args)

    listPaths = []
    numberObject = 24
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
            'channelType':  args.channelType
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
    
    if strtobool(args.eaNorm):
        dataLink = dataName + '_COV.txt'   
        if not os.path.exists(dataLink):
            normR = getNormR(X_train)
            normR = sqrtm(normR)
            normR = np.linalg.inv(normR)
            np.savetxt(dataLink, normR, fmt= '%.3f')
        else:
            normR = np.loadtxt(dataLink)

        for ii in range(len(X_train)):
            Xnew = np.matmul(normR, X_train[ii])
            tmp.append(Xnew)
        X_train = np.asarray(tmp)
        tmp = []
        for ii in range(len(X_test)):
            Xnew = np.matmul(normR, X_test[ii])
            tmp.append(Xnew)
        X_test = np.asarray(tmp)

    # model PSD + SVM
    fft_rs, freq = GetFFT(X_train)
    newXTrain = GetPSD(fft_rs)
    newXTrain = np.asarray(newXTrain)
    newXTrain = newXTrain.real
    newXTrain = newXTrain.reshape(len(newXTrain), -1)
    fft_rs, freq = GetFFT(X_test)
    newXTest = GetPSD(fft_rs)
    newXTest = np.asarray(newXTest)
    newXTest = newXTest.real
    newXTest = newXTest.reshape(len(newXTest), -1)
    
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
    clf.fit(newXTrain, y_train)
    predicted = clf.predict(newXTest)
    predicted = np.asarray(predicted)
    counter = 0 
    for i in range(len(predicted)):
        if predicted[i] == y_test[i]:
            counter += 1
    print(counter * 100.0 / len(predicted))
    stop

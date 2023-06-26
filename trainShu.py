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
import scipy.io as sio

from util.dataUtil import *
from util.modelUtil import *
from util.preproc import *
from os.path import join

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

channelCombos = [
    ['Fz', 'Fp1', 'F7', 'F3', 'FC1', 'FC5', 'FC6', 'FC2', 'F4', 'F8', 'Fp2'],
    ['Cz', 'C3', 'CP5', 'CP1', 'CP2', 'CP6', 'C4'],
    ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"]
]

listChns = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"]

listMethods = ['PSD + SVM', 'IHAR + SVM']
event_id = {'left': 1,'right':2}

def get_dataShu(subj,session,data_path):
    da=sio.loadmat(join(data_path,'sub-'+str(subj).zfill(3)+'_ses-'+str(session).zfill(2)+'_task_motorimagery_eeg.mat'))
    data=da['data']
    labels=np.ravel(da['labels'])
    return data,labels

def mnebandFilter(data,labels,dataInfo):
    si,sj,sk=data.shape
    da=data.transpose(1,0,2)
    da=da.reshape(sj,si*sk)
    llen=data.shape[0]
    event=np.zeros((llen,3))
    info = mne.create_info(
        ch_names=listChns,
        ch_types="eeg",  # channel type
        sfreq= 128  # frequency
    )
    raw = mne.io.RawArray(da, info)  # create raw
    raw.filter(dataInfo['bandL'], dataInfo['bandR'], fir_design='firwin')
    for i in range(llen):
        event[i,0]=i*sk
        event[i,2]=labels[i]
    event=event.astype(int)
    train_epoches = mne.Epochs(raw, event, event_id, 1, 3 + 0.004,
                               baseline=None, preload=True)
    train_data = train_epoches.get_data()
    tmp = np.asarray(event)
    train_data = train_data[np.where(tmp[:,2] == 2)]
    return train_data


def extractDataShu_byInfo(info):
    # info['numSub']
    listData = []
    dss = [0,1,2,3]
    for idx in range(info['numSub']):
        subData = []
        for ts in range(len(dss)):
            ss = dss[ts]
            data, label = get_dataShu(idx+1, ss+1, info['input'])
            train_data  = mnebandFilter(data, label, info)
            transposeData = [x.T for x in train_data]
            subData.append([ts, transposeData])
        listData.append(subData)
    return listData

def trainCore(X_train, X_test, y_train, y_test, info):
    X_train, X_test, _, _ = normMat(X_train, X_test)

    if args.eaNorm == 'DEA':    
        allMat = listRepresent(X_train, y_train, False)
        UmeanMat = getV_SVD(allMat)
        
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

    if args.modelName == 'SVM':
        listData = []
        listLb = []
        for ii in range(len(X_train)):
            if y_train[ii] not in listLb:
                listLb.append(y_train[ii])  
                listData.append([])
            index = listLb.index(y_train[ii])
            listData[index].append(X_train[ii])


        for ii in range(5):
            a = listData[ii]
            plt.imshow(a, cmap='hot', interpolation='nearest')
            plt.savefig(f"{ii}.png")


        listData = []
        listLb = []
        for ii in range(len(X_test)):
            if y_test[ii] not in listLb:
                listLb.append(y_test[ii])  
                listData.append([])
            index = listLb.index(y_test[ii])
            listData[index].append(X_test[ii])


        for ii in range(5):
            a = listData[ii]
            plt.imshow(a, cmap='hot', interpolation='nearest')
            plt.savefig(f"{ii}_test.png")

        print(X_train.shape, X_test.shape)
        return SVM(X_train, y_train, X_test, y_test)

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
        model = chooseModel(str(args.modelName), num_class=num_class+1, input_size=(1, X_train.shape[2], X_train.shape[3]))
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
    parser.add_argument('--channelType', help='channel seclection in : {}'.format(channelCombos), default=2, type=int)
    parser.add_argument('--windowSize', help='windowSize', default=128, type=int)
    parser.add_argument('--trainTestSession', help='train test are splitted by session', default='False')
    parser.add_argument('--output', help='train test are splitted by session', default='./result.txt')
    parser.add_argument('--numSub', help='number of Subject', default=25, type=int)
    parser.add_argument('--numChan', help='number of channel', default=-1, type=int)
    parser.add_argument('--modelFeatures', help='name of features : PSD, IHAR, APF, RAW', default='RAW')
    args = parser.parse_args()
    print(args)
    listPaths = []
    numberObject = args.numSub
    counter = 0


    prePath = args.input
    for x in os.listdir(prePath):
        if x[-3:] == 'mat':
            listPaths.append(prePath + '/' + x)
            counter += 1
            if counter > numberObject:
                break

    tmpExtract = 'All'
    typeTest = 'trainTestRandom'
    if strtobool(args.trainTestSession):
        typeTest = 'trainTestSession'

    dataName = f'./SHU_numberSub{str(args.numSub)}_band_{str(args.bandL)}_{str(args.bandR)}_channelType_{str(args.channelType)}_{typeTest}_{args.windowSize}'
    dataLink = dataName + '.npy'
    # print(dataLink)

    info = {
            'bandL': float(args.bandL),
            'bandR': float(args.bandR),
            'windowSize': args.windowSize,
            'EA': str(args.eaNorm),
            'channelType': channelCombos[args.channelType],
            'modelName': args.modelName, 
            'typeTest': typeTest,
            'numSub': args.numSub, 
            'input': args.input,
            'dataset': 'shu',
            'numChan': args.numChan
        }
    if not os.path.exists(dataLink):
        # normal
        datas = extractDataShu_byInfo(info)
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
        elif typeTest == 'trainTestSession':
            print("Training at {} round".format(testingTime))
            for scenario in range(4):
                X_train, y_train, X_test, y_test = getDataScenario(PreProDatas, scenario)
                acc = trainCore(X_train, X_test, y_train, y_test, info)
                # acc = trainCore(X_test, X_train, y_test, y_train, info)
                print("Scenario {} with acc: {}".format(scenario, acc))
                listAcc.append(acc)
            break

    listAcc = np.asarray(listAcc)
    sourceFile = open(args.output, 'a')
    print('*'*10, 'Result' ,'*'*10, file = sourceFile)
    print(args, file = sourceFile)
    print(listAcc, file = sourceFile)
    print(np.mean(listAcc), np.max(listAcc) - np.mean(listAcc), file = sourceFile)
    print('*'*10, 'End' ,'*'*10, file = sourceFile)
    sourceFile.close()

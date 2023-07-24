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


from braindecode.models import EEGITNet, EEGInception, ShallowFBCSPNet
from skorch.callbacks import LRScheduler

from braindecode import EEGClassifier
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

channelCombos = [
    ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5' ,'CP1', 'CP2', 'CP6'],
    ['O1', 'Oz', 'O2', 'P7', 'P3', 'Pz', 'P4', 'P8'],
    ['O1', 'Oz', 'O2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'CP5', 'CP1', 'CP2', 'CP6', 'T7', 'T8', 'C3', 'C4', 'Cz', 'FT9', 'FT10', 'FC5', 'FC6', 'FC1', 'FC2'],
    ['T7', 'C3', 'Cz', 'C4', 'T8', 'CP5' ,'CP1', 'CP2', 'CP6'],
    ['FC5', 'FC1', 'FC2' ,'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8'],
    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
     'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
]

# channelCombos = [
#     ['Fz', 'Fp1', 'F7', 'F3', 'FC1', 'FC5', 'FT9', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2'],
#     ['P3', 'P7', 'PO9', 'Pz', 'PO10', 'P8', 'P4'],
#     ['Cz', 'C3', 'CP5', 'CP1', 'CP2', 'CP6', 'C4'],
#     ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
#      'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'],
#      ['T7', 'T8']
# ]

listChns = ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8',
     'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']

listMethods = ['PSD + SVM', 'IHAR + SVM']

def mva_signal(signal, size):
    index = 0
    moving_averages = []
    # Loop through the array t o
    #consider every window of size 3
    while index < len(signal) - size + 1:
        # Calculate the average of current window
        window_average = round(np.sum(signal[
          index:index+size]) / size, 2)
        if window_average < 0.00001:
            window_average = 0.00001
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        index += 1
    return np.asarray(moving_averages)

def mva(signals, size):
    newSg = []
    for ii in range(len(signals)):
        newSg.append(mva_signal(signals[ii], size))
    return np.vstack(newSg)

def smooth(trials, size):
    newTrials = []
    for ii in tqdm(range(len(trials))):
        newTrials.append(mva(trials[ii], size))
    return np.asarray(newTrials)

def trainCore(X_train, X_test, y_train, y_test, info):
    X_train, X_test, _, _ = normMat(X_train, X_test)

    if args.eaNorm == 'DEA':    
        allMat = listRepresent(X_train, y_train, True)
        UmeanMat = getV_SVD(allMat)
        
        X_train = transformMat(X_train, UmeanMat, True)
        X_test = transformMat(X_test, UmeanMat, True)

        X_train, X_test = EANorm(X_train, X_test)

    elif args.eaNorm == 'EA':
        X_train, X_test = EANorm(X_train, X_test, X_train)

    if args.modelFeatures == 'PSD':
        X_train, y_train, X_test, y_test = PSD(X_train, y_train, X_test, y_test)
    elif args.modelFeatures == 'IHAR':
        X_train, y_train, X_test, y_test = IHAR(X_train, y_train, X_test, y_test, listChns)           
    elif args.modelFeatures == 'APF':
        X_train = np.mean(np.log(np.abs(smooth(X_train, info['deltaSize']))) , axis = 1)
        X_test = np.mean(np.log(np.abs(smooth(X_test, info['deltaSize']))), axis = 1)

    if args.modelName == 'SVM':
        return SVM(X_train, y_train, X_test, y_test)
    elif args.modelName in['ITNET', 'FBCSP', 'INCEPTION']:
        n_classes = len(np.unique(y_train))
        n_channels= X_train.shape[-1]
        input_window_samples = X_train.shape[1]
        X_test = np.transpose(X_test, (0, 2, 1))
        X_train = np.transpose(X_train, (0, 2, 1))
        if args.modelName == 'ITNET':
            # n_samples, n_timestamp, n_channels = X_train.shape
            # X_train = X_train.reshape((n_samples, n_timestamp, n_channels, 1))

            # n_samples, n_timestamp, n_channels = X_test.shape
            # X_test = X_test.reshape((n_samples, n_timestamp, n_channels, 1))
            model = EEGITNet(n_classes,
                        n_channels,
                        input_window_samples=input_window_samples)
        elif args.modelName == 'FBCSP':
            model = ShallowFBCSPNet(n_channels,
                        n_classes,
                        input_window_samples=input_window_samples, final_conv_length=6, pool_time_length=25)
        else:
            model = EEGInception(n_channels,
                        n_classes,
                        input_window_samples=input_window_samples)

        model.to(device)
        lr = 7e-2
        weight_decay = 1e-5
        batch_size = 64
        n_epochs = 20
        clf = EEGClassifier(
                            model,
                            criterion=torch.nn.CrossEntropyLoss,
                            optimizer=torch.optim.Adam,
                            train_split=None,
                            optimizer__lr=lr,
                            optimizer__weight_decay=weight_decay,
                            batch_size=batch_size,
                            callbacks=[
                                "accuracy",
                                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
                            ],
                            device=device,
                        )
        clf.fit(X_train, y=np.asarray(y_train), epochs=n_epochs)

        # score the Model after training
        test_acc = clf.score(X_test, y=np.asarray(y_test))
        print(f"Test acc: {(test_acc * 100):.2f}%") 
        return test_acc
    elif (info['modelName'] in ['CNN', 'CNN_LSTM']):
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
    parser.add_argument('--input', help='input data dir', default='./Official')
    parser.add_argument('--modelName', help='name of model : {}'.format(listMethods))
    parser.add_argument('--modelFeatures', help='name of features : PSD, IHAR, APF, RAW', default='RAW')
    parser.add_argument('--bandL', help='band filter', default=4.0, type=float)
    parser.add_argument('--bandR', help='band filter', default=50.0, type=float)
    parser.add_argument('--eaNorm', help='EA norm', default='False')
    parser.add_argument('--channelType', help='channel seclection in : {}'.format(channelCombos), default=3, type=int)
    parser.add_argument('--windowSize', help='windowSize', default=128, type=int)
    parser.add_argument('--deltaSize', help='deltaSize', default=1, type=int)
    parser.add_argument('--windowIHAR', help='windowIHAR', default=10, type=int)
    parser.add_argument('--extractFixation', help='type of extraction in eeg. Fixation: True. All: False', default='False')
    parser.add_argument('--thinking', help='thinking: True. resting: False', default='False')
    parser.add_argument('--trainTestSeperate', help='train first then test. if not, train and test are splitted randomly', default='False')
    parser.add_argument('--trainTestSession', help='train test are splitted by session', default='False')
    parser.add_argument('--naiveClss', help='name of primitive classifier : {}'.format([
        'SVM_Linear', 'SVM_RBF', 'NearestNeighbor', 'NaiveBayes', 'RF', 'GaussianProcess', 'simpleNeuralNet']))
    parser.add_argument('--output', help='train test are splitted by session', default='./result.txt')
    args = parser.parse_args()
    print(args)
    '''
    # python train.py --windowSize 128 --modelName PSD --bandL 0.1 --bandR 50 --extractFixation False --thinking False --trainTestSeperate False --trainTestSession False
    '''
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
            'dataset': None,
            'deltaSize': args.deltaSize,
            'naiveClss': args.naiveClss,
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
    print(typeTest)
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

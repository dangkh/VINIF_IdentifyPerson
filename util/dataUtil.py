from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from util.nets import *
import json
import mne
import pandas as pd
import sys
from scipy.linalg import sqrtm
from sklearn.metrics import confusion_matrix

def chunk(matrix, size):
    list_matrix = []

    l, r = matrix.shape[0]-1 - size, matrix.shape[0]-1
    
    while l >= 0:
        subMatrix = np.copy(matrix[l:r])
        list_matrix.append(subMatrix)
        l -= size
        r -= size
    return list_matrix

def chunk_matrix(list_data, label, size):
    list_matries = []
    list_ys = []
    for idx, matrix in enumerate(list_data):
        matries = chunk(matrix, size)
        list_matries.extend(matries)
    y_matries = [label] * len(list_matries)

    return list_matries, y_matries


def preprocessDataInfo(inputData, info):
    result = []
    numberSub = len(inputData)
    for subId in range(numberSub):
        numberSample = len(inputData[subId])
        subData = inputData[subId]
        preprocSub = []
        for sampleId in range(numberSample):
            scenarioId, listEEG, _ = subData[sampleId]
            # print(scenarioId)
            newData, label = chunk_matrix(listEEG, scenarioId, size=int(info['windowSize']))
            preprocSub.append([newData, label])
            # break
        result.append(preprocSub)
    """
    result comprises [dataSub1,dataSub2...]
    dataSubx comprises [[data, scenarioID], [data, scenarioID]]
    """
    return result


def preprocessData(inputData, size):
    result = []
    
    numberSub = len(inputData)
    for subId in range(numberSub):
        numberSample = len(inputData[subId])
        subData = inputData[subId]
        preprocSub = []
        for sampleId in range(numberSample):
            scenarioId, listEEG, listFixation = subData[sampleId]
            # print(scenarioId)
            newData, label = chunk_matrix(listEEG, scenarioId, size = size)
            preprocSub.append([newData, label])
            # break
        result.append(preprocSub)
    """
    result comprises [[dataSub1],[dataSub2]...]
    dataSubx comprises [[data, scenarioID], [data, scenarioID]]
    """
    return result


def preprocessDataMI(inputData, size):
    result = []
    
    numberSub = len(inputData)
    for subId in range(numberSub):
        numberSample = len(inputData[subId])
        subData = inputData[subId]
        preprocSub = []
        for sampleId in range(numberSample):
            scenarioId, listEEG = subData[sampleId]
            # print(scenarioId)
            newData, label = chunk_matrix(listEEG, scenarioId, size = size)
            preprocSub.append([newData, label])
            # break
        result.append(preprocSub)
    """
    result comprises [[dataSub1],[dataSub2]...]
    dataSubx comprises [[data, scenarioID], [data, scenarioID]]
    """
    return result

def analyzeTrainData(Ys, title = "chart number"):
    (unique, counts) = np.unique(np.asarray(Ys), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    # print(frequencies)
    plt.title(title)
    plt.bar(unique, counts)
    plt.show()

def analyze(inputData):
    Ys = []
    for id, dataSub in enumerate(inputData):
        for _, samples in dataSub:
            label = [id] * len(samples)
            Ys.extend(label)
    analyzeTrainData(Ys, "chart by Subject")

    Ys = []
    for id, dataSub in enumerate(inputData):
        for _, samples in dataSub:
            Ys.extend(samples)
    analyzeTrainData(Ys, "chart by Scenario")

def analyzeSub(inputData, subId):
    Ys = []
    for _, samples in inputData:
        Ys.extend(samples)
    analyzeTrainData(Ys, "chart by Subject: " + str(subId))

class EEG_data(Dataset):
    def __init__(self, datas, targets=None,
                 train=True):

        self.y = targets
        # for SHALLOW
        # meanMat = np.mean(datas, axis=1, keepdims=True)
        # stdMat = np.std(datas, axis=1, keepdims=True)
        
        # for Normal
        # meanMat = np.mean(datas, axis=3, keepdims=True)
        # stdMat = np.std(datas, axis=3, keepdims=True)
        
        # self.X = (datas - meanMat) / stdMat
        
        # for DEA
        self.X = np.asarray(datas)
        self.X = self.X.astype(np.float32) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.tensor(self.X[idx]), self.y[idx]
        else:
            return torch.tensor(self.X[idx])

def TrainTestLoader(data, testSize = 0.1, split_augment = []):
    if len(data) == 2:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=testSize, random_state=42)
        if len(split_augment) > 0:
            X_train = np.transpose(X_train, (0, 2, 3, 1))
            print("number sample training :", len(y_train))   
            X_train, y_train = augmentData(X_train, y_train, labels = split_augment)
            print("number sample training after augmented :", len(y_train))      
            X_train = np.transpose(X_train, (0, 3, 1, 2))
    else:
        [X_train, y_train, X_test, y_test] = data
    batch_size = 32
    train_dataset = EEG_data(X_train, y_train)
    test_dataset = EEG_data(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= batch_size)

    return train_loader, test_loader

def trainTestSplit(data, target, test, testIndex):
    X_train = []
    X_test= []
    y_train = []
    y_test = []
    for idx in range(len(data)):
        if test[idx] == testIndex:
            X_test.append(data[idx])
            y_test.append(target[idx])
        else:
            X_train.append(data[idx])
            y_train.append(target[idx])
    return X_train, y_train, X_test, y_test


"""
    inputData comprises [[dataSub1],[dataSub2]...]
    dataSubx comprises [[data, scenarioID], [data, scenarioID]]
"""
# shuffle and random split
def getDataRandom(inputData):
    data, label = [], []
    for id, dataSub in enumerate(inputData):
        for dataSample, _ in dataSub:
            subLabel = [id] * len(dataSample)
            data.extend(dataSample)
            label.extend(subLabel)
    return np.asarray(data), label
# split by scenario

def getDataScenario(inputData, testIndex):
    data, label, test = [], [], []
    for id, dataSub in enumerate(inputData):
        for dataSample, scenarioId in dataSub:
            subLabel = [id] * len(dataSample)
            data.extend(dataSample)
            label.extend(subLabel)
            testLabel = []
            if len(scenarioId) > 0 and scenarioId[0] == testIndex:
                testLabel = [1] * len(dataSample)
                test.extend(testLabel)
            else:
                testLabel = [0] * len(dataSample)
                test.extend(testLabel)
    X_train, y_train, X_test, y_test = trainTestSplit(data, label, test, 1)
    return np.asarray(X_train), y_train, np.asarray(X_test), y_test

# split by session
def getDataFuture(inputData, info):
    data, label = [], []
    for id, dataSub in enumerate(inputData):
        for dataSample, _ in dataSub:
            data.extend(dataSample)
            label.extend([id]*len(dataSample))
    data = np.asarray(data)
    label = np.asarray(label)
    ids = np.unique(label)
    # extract by name
    ratio = 0.1
    train, test, keys = splitLabel(label, ratio, info)
    X_train, X_test, y_train, y_test = [], [], [], []
    for sampleIdx in range(len(data)):
        if label[sampleIdx] in train:
            X_train.append(data[sampleIdx])
            y_train.append(keys[label[sampleIdx]])
        else:
            X_test.append(data[sampleIdx])
            y_test.append(keys[label[sampleIdx]])
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)


def findSub(info):
    '''
    return 2D list, [[s1],[s2],...,[sn]]
    where si contain [label1, label2, label3,...,labeln] 
    '''
    listPaths = info['listPaths']
    list_dir = []
    for dir in listPaths:
        if dir[-9:] == '.DS_Store':
            continue
        if not os.path.isdir(dir):
            continue
        tmp = dir.split('/')
        list_dir.append(tmp[-1])

    listSub = [[0]]
    for ii in range(1, len(list_dir)):
        if list_dir[ii][:7] == list_dir[ii-1][:7]:
            listSub[-1].append(ii)
        else:
            listSub.append([ii])
    return listSub


def randomSelect(sub, numTesting):
    tmp = sub
    np.random.shuffle(tmp)

    train = tmp[:-numTesting]
    test = tmp[-numTesting:]
    print("*"*10,"info","*"*10)

    print(train)
    print(test)
    print("*"*10,"end","*"*10)
    return train, test


def splitLabel(label, ratio, info):
    '''
    return 2 list of label, are radomly selected as given ratio
    '''

    labelKey  =  [0]*(len(np.unique(label))+1)
    totalTrain, totalTest = [], []
    listSub = findSub(info)

    for idx in range(len(listSub)):
        for X in listSub[idx]:
            labelKey[X] = idx
    for sub in range(len(listSub)):
        # numTesting = max(int(len(listSub[sub]) * ratio), 1)
        numTesting = 1
        # random select testing sample
        train, test = randomSelect(listSub[sub], numTesting)
        totalTrain.extend(train)
        totalTest.extend(test)
    return totalTrain, totalTest, labelKey


# get MI data all of it
def getData_All(inputData):
    data, label = [], []
    for id, dataSub in enumerate(inputData):
        for dataSample, _ in dataSub:
            data.extend(dataSample)
            label.extend([id]*len(dataSample))
    return np.asarray(data), np.asarray(label)


def getDataMI_All_byPerson(inputData, personID = 0, personList = 0):
    data, label = [], []
    # for id, dataSub in enumerate(inputData[personID]):
    id = personID
    dataSub = inputData[personID]
    for dataSample, _ in dataSub:
        # tmp = 0
        # if id > 10: 
        #     tmp = 1
        data.extend(dataSample)
        label.extend([id]*len(dataSample))
    return np.asarray(data), np.asarray(label)

def addNoise(data, target):
    list_newdata = []
    list_newtarget = []
    tmp = np.asarray(target)
    mean = np.mean(tmp)
    for idx in range(len(data)):
        tmpTarget = [0] * 12
        matrix = np.copy(data[idx])
        noise = np.random.normal(mean, 0.1, size=matrix.shape)
        newmatrix = matrix + noise
        newmatrix = newmatrix.astype(np.double)
        list_newdata.append(newmatrix)
        # tmpTarget[target[idx]] = 1
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget


def randomRemoveSample(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        matrix = np.copy(data[idx])
        lenRandom = matrix.shape[0]
        numRandom = int(lenRandom / 10)
        listFrame = []
        for id in range(numRandom):
            tmp = np.random.randint(1, matrix.shape[0])
            listFrame.append(tmp)
        for f in listFrame:
            if f > 1 and f < lenRandom - 1:
                matrix[f] = matrix[f - 1] + matrix[f + 1] / 2
        # print(matrix.shape)
        list_newdata.append(matrix)
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget


def randomSwapSample(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        matrix = np.copy(data[idx])
        lenRandom = matrix.shape[0]
        numRandom = 8
        listFrame = []
        for id in range(numRandom):
            tmp = np.random.randint(3, lenRandom - 10)
            listFrame.append(tmp)
        listFrame = np.sort(listFrame)
        for idy, v in enumerate(listFrame):
            if idy > 0 and listFrame[idy] < listFrame[idy - 1]:
                listFrame[idy] += 1

        list_Matrix = []
        for x in range(4):
            l = x * 2
            r = x * 2 + 1
            dl = listFrame[l]
            dr = listFrame[r]
            tmpMatrix = np.copy(matrix[dl:dr])
            list_Matrix.append(tmpMatrix)
        swapMatrix = []
        arr = np.arange(4)
        np.random.shuffle(arr)
        for x in range(4):
            swapMatrix.append(np.copy(list_Matrix[arr[x]]))

        listFrame = np.insert(listFrame, 0, 0)
        listFrame = np.append(listFrame, lenRandom)
        list_Matrix = []
        for x in range(5):
            l = x * 2
            r = x * 2 + 1
            dl = listFrame[l]
            dr = listFrame[r]
            tmpMatrix = np.copy(matrix[dl:dr])
            list_Matrix.append(tmpMatrix)

        finalList = []
        for x in range(4):
            finalList.append(list_Matrix[x])
            finalList.append(swapMatrix[x])
        finalList.append(list_Matrix[-1])

        finalMatrix = np.vstack(finalList)

        list_newdata.append(finalMatrix)
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget


def augmentData(Xs, Ys, labels):
    newXs = []
    newYs = []
    for label in labels:
        X_source = Xs[np.where(Ys == label)]
        y_source = Ys[np.where(Ys == label)]
        datanoise, targetnoise = addNoise(X_source, y_source)
        # dataRemove, targetRemove = randomRemoveSample(X_source, y_source)
        dataSwap, targetSwap = randomSwapSample(X_source, y_source)
        newXs.extend(datanoise)
        # newXs.extend(dataRemove)
        newXs.extend(dataSwap)
        newYs.extend(targetnoise)
        # newYs.extend(targetRemove)
        newYs.extend(targetSwap)
    newXs.extend(Xs)
    newYs.extend(Ys)
    return np.asarray(newXs), np.asarray(newYs)

def extractData_byInfo(info):
    datas = []
    listPaths = info['listPaths']
    list_dir = []
    for sub in listPaths:
        list_dir.append(sub)

    # print(list_dir)
    for dir in list_dir:
        if dir[-9:] == '.DS_Store':
            continue
        if not os.path.isdir(dir):
            continue
        data = extractSub_byInfo(dir, info)
        datas.append(data)

    return datas

def ET2Fixation(etFile):
    print(etFile.head())
    # charactertyping
    n = len(etFile['sentence'])
    print("Time in ET: ", n / 60, "*" * 10)
    startTyping = 0
    stopTyping = n
    for i in range(n - 1):
        if etFile['sentence'][i] != "MainMenu":
            startTyping = i
            break
    for i in range(n - 1, 0, -1):
        if etFile['sentence'][i] != "MainMenu":
            stopTyping = i
            break

    # print(startTyping, " ", stopTyping)
    lastcharacter = startTyping + 1
    listFix = []
    for i in range(startTyping + 2, stopTyping - 1):
        if str(etFile['character typing'][lastcharacter]) != str(etFile['character typing'][i]):
            tmp = abs(etFile['TimeStamp'][lastcharacter] - etFile['TimeStamp'][i - 1])
            if tmp >= 1.4 and tmp <= 1.7:
                listFix.append([etFile['TimeStamp'][lastcharacter], etFile['TimeStamp'][i - 1]])
                print(etFile['character typing'][lastcharacter], etFile['character typing']
                      [i], (etFile['TimeStamp'][lastcharacter] - etFile['TimeStamp'][i - 1]))
            lastcharacter = i
    # print(listFix)
    return listFix


def extractSub_byInfo(path, info):
    samples = os.listdir(path)
    # print(samples)
    data = []
    for idx, sample in enumerate(samples):
        samplePath = path + '/' + sample + '/'
        jsonpath = samplePath + "scenario.json"
        if os.path.isdir(samplePath):
            print(samplePath)
            if os.path.isfile(samplePath + "/cleanET.csv"):
                et_raw = pd.read_csv(samplePath + "/cleanET.csv")
            else:
                et_raw = pd.read_csv(samplePath + "/ET.csv")
            print('*'*8,"Extracting Fixation", '*'*8)
            listFixation = ET2Fixation(et_raw)
            print('*'*8,"Extracting Fixation Finished", '*'*8)

            scenarioNum = -1
            with open(jsonpath) as json_file:
                datajs = json.load(json_file)
                scenarioNum = getScenNumber(datajs)

            if info['extractFixation']:
                listEEG, _ = EEGByFixation_byInfo(samplePath, listFixation, info)
            else:
                listEEG, _ = EEGExtractor_byInfo(samplePath, info)
            data.append([scenarioNum - 1, listEEG, ''])
    return data


def getScenNumber(jsfile):
    # return scenario Number rgds to input jsfile
    res = -1
    listkey = ["scenarioId", "RecPlanEdit", "scenarioNumber"]
    for key in listkey:
        try:
            res = jsfile[key]
        except Exception as e:
            print('error key' + key)
    return res

def checkSubFolder(path):
    # return list folder contain data according to input path
    list_dir = []
    subs = os.listdir(path)
    for sub in subs:
        subPath = path + '/' + sub
        infoPath = path + '/' + sub + '/info.json'
        if os.path.isfile(infoPath) and os.path.isdir(subPath):
            list_dir.append(subPath)
    return list_dir


def EEGExtractor_byInfo(link, info):
    eeg_raw = mne.io.read_raw_edf(link + "/EEG.edf")
    print(eeg_raw.info)
    eeg_data_new = eeg_raw.copy().load_data().filter(l_freq= float(info['bandL']), h_freq= float(info['bandR']))
    eegTs = pd.read_csv(link + '/EEGTimeStamp.txt', names=['TimeStamp'])
    eegTs = eegTs.sort_values(by=['TimeStamp'])
    print("Time in EEGTS by Frame number: ", len(eegTs) / 128, "*" * 20)
    print("Time in EEGTS by TS: ", eegTs["TimeStamp"].iloc[-1] - eegTs["TimeStamp"].iloc[0], "*" * 20)
    timeInTs = len(eegTs) / 128
    timeInFrame = eegTs["TimeStamp"].iloc[-1] - eegTs["TimeStamp"].iloc[0]
    if abs(timeInTs - timeInFrame) > 2:
        return [], []
    print(eeg_data_new.annotations)
    listEEG = []
    compareArea = "Resting"
    if info['thinking']:
        compareArea = "Thinking"
    for annos in eeg_data_new.annotations:
        if annos["description"] != compareArea:
            continue
        start = annos["onset"] + 1.1
        y = annos["duration"]
        if y < 1.1:
            continue
        stop = start + y - 1.1
        tmp = eeg_data_new.copy().crop(start, stop)
        matrix = tmp.get_data(picks = info['channelType']).T

        # uncomment neu chon toan bo channel
        # tmp = tmp.to_data_frame()
        # matrix = tmp.iloc[:, 1:].to_numpy()
        listEEG.append(matrix)
    return listEEG, ""

def EEGByFixation_byInfo(link, listFixation, info):
    eeg_raw = mne.io.read_raw_edf(link + "/EEG.edf")
    print(eeg_raw.info)
    eeg_data_new = eeg_raw.copy().load_data().filter(l_freq= float(info['bandL']), h_freq= float(info['bandR']))
    eegTs = pd.read_csv(link + '/EEGTimeStamp.txt', names=['TimeStamp'])
    eegTs = eegTs.sort_values(by=['TimeStamp'])
    print("Time in EEGTS by Frame number: ", len(eegTs) / 128, "*" * 20)
    print("Time in EEGTS by TS: ", eegTs["TimeStamp"].iloc[-1] - eegTs["TimeStamp"].iloc[0], "*" * 20)
    timeInTs = len(eegTs) / 128
    timeInFrame = eegTs["TimeStamp"].iloc[-1] - eegTs["TimeStamp"].iloc[0]
    if abs(timeInTs - timeInFrame) > 2:
        return [], []
    eegStartTs = eegTs["TimeStamp"][0]
    listEEG = []
    listDiff = []

    for idx, rangeFix in enumerate(listFixation):
        startFix, stopFix = listFixation[idx]
        print("Fix info: ", startFix, " ", stopFix, " ", stopFix - startFix)
        if stopFix - startFix <= 0.1:
            continue
        if len(eegTs[eegTs['TimeStamp'] >= startFix]) == 0:
            break
        # get EEG frame by index
        start = [eegTs[eegTs['TimeStamp'] >= startFix].index[0] /  128 + 0.01]
        # print(eegTs[eegTs['TimeStamp'] <= stopFix])
        stop = [eegTs[eegTs['TimeStamp'] <= stopFix].index[-1] /  128]
        # print(start," " ,stop, " ", stop - start )

        diffTs = (stopFix - startFix) - (stop[0] - start[0])
        # print(diffTs)

        if start[0] < stop[0]:
            try:
                tmp = eeg_data_new.copy().crop(start[0], stop[0])
                matrix = tmp.get_data(picks = info['channelType']).T
                numFrame = matrix.shape[0]
                numFrame2time = numFrame /  128
                diffEEGvET = (stopFix - startFix) - numFrame2time
                diffEEGvTs = (stop[0] - start[0]) - numFrame2time
                # print(diffEEGvET, diffEEGvTs)
                listDiff.append([diffTs, diffEEGvET, diffEEGvTs])
                listEEG.append(matrix)
            except Exception as e:
                print("error")
    return listEEG, ''

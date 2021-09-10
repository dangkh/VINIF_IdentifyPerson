from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import numpy as np

from sklearn.model_selection import train_test_split

def chunk(matrix, step_size=128, window_size=128):
    list_matrix = []
    l, r = 0, window_size - 1
    while r <= matrix.shape[0]:
        subMatrix = np.copy(matrix[l:r])
        list_matrix.append(subMatrix)
        l += step_size
        r += step_size
    l, r = matrix.shape[0] - window_size, matrix.shape[0] - 1
    subMatrix = np.abs(np.copy(matrix[l:r]))
    subMatrix = subMatrix.astype(np.double)
    list_matrix.append(subMatrix)
    return list_matrix


def chunk_matrix(list_data, list_target, list_test, step_size=128, window_size=128):
    list_matries = []
    list_ys = []
    list_t = []
    for idx, matrix in enumerate(list_data):
        matries = chunk(matrix)
        list_matries.extend(matries)
        y_matries = [list_target[idx]] * len(matries)
        list_ys.extend(y_matries)
        test_matries = [list_test[idx]] * len(matries)
        list_t.extend(test_matries)

    return list_matries, list_ys, list_t


def addNoise(data, target, test):
    list_newdata = []
    list_newtarget = []
    list_newtest = []
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
        list_newtest.append(test[idx])
    return list_newdata, list_newtarget, list_newtest


def randomRemoveSample(data, target, test):
    list_newdata = []
    list_newtarget = []
    list_newtest = []
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
        list_newtest.append(test[idx])
    return list_newdata, list_newtarget, list_newtest


def randomSwapSample(data, target, test):
    list_newdata = []
    list_newtarget = []
    list_newtest = []
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
        list_newtest.append(test[idx])
    return list_newdata, list_newtarget, list_newtest


def augmentData(Xs, Ys, Ts, labels = None):
    newXs = []
    newYs = []
    newTs = []
    augmentLabel = labels
    if labels is None:
        augmentLabel = np.unique(Ys)
    for label in augmentLabel:
        X_source = Xs[np.where(Ys == label)]
        y_source = Ys[np.where(Ys == label)]
        t_source = Ts[np.where(Ys == label)]
        datanoise, targetnoise, testnoise = addNoise(X_source, y_source, t_source)
        dataRemove, targetRemove, testRemove = randomRemoveSample(X_source, y_source, t_source)
        dataSwap, targetSwap, testSwap = randomSwapSample(X_source, y_source, t_source)
        newXs.extend(datanoise)
        newXs.extend(dataRemove)
        newXs.extend(dataSwap)
        newYs.extend(targetnoise)
        newYs.extend(targetRemove)
        newYs.extend(targetSwap)
        newTs.extend(testnoise)
        newTs.extend(testRemove)
        newTs.extend(testSwap)
    newXs.extend(Xs)
    newYs.extend(Ys)
    newTs.extend(Ts)
    return np.asarray(newXs), np.asarray(newYs), np.asarray(newTs)


def preprocessData(data, augment = True):
    # augment data & chunk matrix & relabel
    # datas contains data, targets contains label, data_indexes contains 
    # scenario index of phase index
    targets = []
    datas = []
    data_indexes = []
    for idx, _ in enumerate(data):
        info = data[idx]
        targets.append(info[0])
        datas.append(info[1])
        data_indexes.append(info[2])
    newdata, newtarget, newtest = chunk_matrix(datas, targets, data_indexes)
    datas = np.asarray(newdata)
    targets = np.asarray(newtarget)
    data_indexes = np.asarray(newtest)
    if augment:
        newdata, newtarget, newtest = augmentData(datas, targets, data_indexes)
        datas = np.asarray(newdata)
        targets = np.asarray(newtarget)
        data_indexes = np.asarray(newtest)

    # relabel
    (unique, counts) = np.unique(np.asarray(targets), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)

    # missing at some label, relabel from 1 - number of unique label
    counter = 0
    list_label = [x for x in range(len(frequencies))]
    newtargets = np.copy(targets)
    for x in range(len(frequencies)):
        newtargets[np.where(targets == unique[x])] = x
    targets = newtargets
    print("new targets")
    (unique, counts) = np.unique(np.asarray(targets), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)
    return datas, targets, data_indexes

class EEG_data(Dataset):
    def __init__(self, datas, targets=None,
                 train=True):

        self.y = targets
        mean = np.mean(datas, axis=3, keepdims=True)
        std = np.std(datas, axis=3, keepdims=True)
        self.X = (datas - mean) / std
        self.X = self.X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.tensor(self.X[idx]), self.y[idx]
        else:
            return torch.tensor(self.X[idx])

def TrainTestLoader(data, testSize=0.1):
    if len(data) == 2:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=testSize, random_state=42)
    else:
        [X_train, y_train, X_test, y_test] = data
    batch_size = 32

    train_dataset = EEG_data(X_train, y_train)
    test_dataset = EEG_data(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

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
    return X_train, X_test, y_train, y_test
        

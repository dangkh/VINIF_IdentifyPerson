from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from scipy.linalg import sqrtm
from util.nets import *
import json
import mne
import pandas as pd
import sys
from scipy.linalg import sqrtm
from sklearn.metrics import confusion_matrix
from util.preproc import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


h1, w1 = 3, 1
h2, w2 = 3, 3
h3, w3 = 3, 5
width = 10
height = 10
paramsCNN2D = {'conv_channels': [
            [1, 16, 8],
            [1, 16, 32],
            [1, 64, 32, 16, 8],
            [1, 128, 64, 32, 16, 8],
            [1, 256, 128, 64, 32, 16, 8]
        ],
            'kernel_size': [[(h1, w1 * width), (h1, w1 * width), (h1, w1 * width),
                             (h1, w1 * width), (h1, w1 * width), (h1, w1 * width)],

                            [(h2 * height, w2), (h2 * height, w2), (h2 * height, w2),
                             (h2 * height, w2), (h2 * height, w2), (h2 * height, w2)],

                            [(h3, w3 * width), (h3, w3 * width), (h3, w3 * width),
                             (h3, w3 * width), (h3, w3 * width), (h3, w3 * width)]]
        }



def chooseModel(modelName, num_class, input_size=None):
    if modelName == "CNN":
        keys = list(paramsCNN2D)
        d = {
            'kernel_size': paramsCNN2D['kernel_size'][1],
            'conv_channels': paramsCNN2D['conv_channels'][1]
        }
        model = CNN2D(  input_size        = input_size,
                        kernel_size     = d['kernel_size'],
                        conv_channels   = d['conv_channels'],
                        dense_size      = 128,
                        dropout         = 0.75, 
                        nclass          = num_class)
    elif modelName == "CNN_LSTM":
        keys = list(paramsCNN2D)
        d = {
            'kernel_size': paramsCNN2D['kernel_size'][1],
            'conv_channels': paramsCNN2D['conv_channels'][1]
        }
        model = CNN_LSTM(input_size    = input_size,
                        kernel_size   = d['kernel_size'],
                        conv_channels = d['conv_channels'],
                        dense_size    = 128,
                        dropout       = 0.75, 
                        nclass = num_class)
    elif modelName == "GCN":
        keys = list(paramsCNN2D)
        d = {
            'kernel_size': paramsCNN2D['kernel_size'][1],
            'conv_channels': paramsCNN2D['conv_channels'][1]
        }
        model = GCN(input_size    = input_size,
                        kernel_size   = d['kernel_size'],
                        conv_channels = d['conv_channels'],
                        dense_size    = 128,
                        dropout       = 0.5, 
                        nclass = num_class)
    else:
        model = WvConvNet(num_class, 6, 2, drop_rate=0.5, flatten=True, input_size = input_size)
        print(model)
    return model

def evaluateModel(model, plotConfusion, dataLoader, n_class):
    counter = 0
    total = 0
    preds = []
    trueLabel = []
    model.to(device)
    model.eval()
    for idx, data in enumerate(dataLoader):
        xx, yy = data
        trueLabel.extend(yy.numpy())
        total += len(yy)
        xx = xx.to(device)
        with torch.no_grad():
            # pred = model(xx, adj)
            pred = model(xx)
            res = torch.argmax(pred, 1)
            if torch.cuda.is_available():
                res = res.cpu().detach()
            preds.extend(res.numpy())
            for id, ypred in enumerate(res):
                if ypred == yy[id].item():
                    counter += 1
    print('acc: {:1f}%'.format(100 * counter / total))
    if plotConfusion:
        plotCl = [str(x) for x in range(n_class)]
        plot_confusion_matrix(trueLabel, preds, classes=plotCl, normalize=True, title='Validation confusion matrix')
    # model.to(device)
    return 100 * counter / total


def trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, n_class, log_batch):
    llos = []
    best_acc = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        model.train()
        print("")
        print("epoch:  {0} / {1}   ".format(epoch, n_epochs))
        running_loss = []
        total_loss = 0
        for i, data in enumerate(trainLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            # CUDA
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # mat = np.asarray(inputs)
            # s1, s2, s3, s4 = mat.shape
            # mat = mat.reshape(s1*s2, s3, s4)
            # coMat = mat.mean(axis = 0)
            # Adj = np.abs(np.corrcoef(coMat[:,:].T))
            # matAdj = torch.Tensor(Adj).to(device)
            # outputs = model(inputs, adj)
            outputs = model(inputs)
            # model CNN LSTM
            loss = criterion(outputs, labels)
            # model Shallow
            # loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += [loss.item()]
            total_loss += loss.item()
            if (i + 1) % log_batch == 0:    # print every 200 mini-batches
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_batch))
                percent = int(i * 50 / len(trainLoader))
                remain = 50 - percent
                sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format(
                    '#' * percent + '-' * remain, percent * 2, np.mean(running_loss)))
                sys.stdout.flush()

                # if (i + 1) / log_batch >= 10:
                #    break

        mean_loss = total_loss / len(trainLoader)
        llos.append(mean_loss)
        scheduler.step()
        sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#' * 50, 100, mean_loss))
        sys.stdout.flush()
        acc = evaluateModel(model, plotConfusion=True, dataLoader=validLoader, n_class=n_class)
        accTrain = evaluateModel(model, plotConfusion=False, dataLoader=trainLoader, n_class=n_class)
    return model, llos, acc, accTrain



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


def IHAR(X_train, y_train, X_test, y_test, listChns):
    
    electrodeIHAR = [['Fp1', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1'], ['Fp2', 'F8', 'F4', 'FC6', 'T8', 'P8', 'O2']]
    numNode = len(electrodeIHAR[0])
    electrodeIndex = []
    for electrodes in electrodeIHAR:
        electrodeIndex.append([listChns.index(x) for x in electrodes])
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
        newX = np.vstack([np.vstack(IharTrain), newX])
        newX = newX.reshape(-1)
        tmp.append(newX)
    X_test = np.vstack(tmp)    
    return SVM(X_train, y_train, X_test, y_test)

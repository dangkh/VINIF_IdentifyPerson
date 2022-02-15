import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mne
from mne import io
from mne.datasets import sample
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time
import os
import json
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import sys
from ultis import *
from nets import *
import pickle


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def checkSubFolder(path):
    # return list folder contain data according to input path
    list_dir = []
    subs = os.listdir(path)
    for sub in subs:
        subPath = path + '/'+ sub
        infoPath = path + '/' + sub + '/info.json'
        if os.path.isfile(infoPath) and os.path.isdir(subPath):
            list_dir.append(subPath)
    return list_dir

def subInVIN():
    # return list folder contain data in VIN project
    list_dir = []
    # list_dir.extend(checkSubFolder("/mnt/hdd/VINIF/DataVIN"))
    # list_dir.extend(checkSubFolder("/mnt/hdd/VINIF/DataVIN/Official"))
    prePath = "../DataVIN/"
    # listPaths = ['HMI09', 'HMI10', 'HMI11', 'HMI13', 'HMI15', 'Official/BN001', 
    #  'Official/BN002', 'Official/K299', 'Official/K300', 'Official/BN003']
    # listPaths = ['HMI10', 'HMI11', 'HMI13', 'HMI15', 'BV103_01', 'Official/BN001', 
    #  'Official/BN002', 'Official/K299', 'Official/K300', 'Official/BN003']
    
    for sub in listPaths:
        list_dir.append(prePath + sub)
    return list_dir

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


def ET2Fixation(etFile):
    print(etFile.head())
    # charactertyping
    n = len(etFile['sentence'])
    print("Time in ET: ", n / 60, "*"*10)
    startTyping = 0
    stopTyping = n
    for i in range(n-1):
        if etFile['sentence'][i] != "MainMenu":
            startTyping = i
            break
    for i in range(n-1,0,-1):
        if etFile['sentence'][i] != "MainMenu":
            stopTyping = i
            break
    
    # print(startTyping, " ", stopTyping)
    lastcharacter = startTyping+1
    listFix = []
    for i in range(startTyping+2, stopTyping-1):
        if str(etFile['character typing'][lastcharacter]) != str(etFile['character typing'][i]):
            tmp = abs(etFile['TimeStamp'][lastcharacter] - etFile['TimeStamp'][i-1])
            if  tmp >= 1.4 and tmp <= 2.0:
                listFix.append([etFile['TimeStamp'][lastcharacter], etFile['TimeStamp'][i-1]])
                print(etFile['character typing'][lastcharacter], etFile['character typing'][i], (etFile['TimeStamp'][lastcharacter] - etFile['TimeStamp'][i-1]))
            lastcharacter = i
    # print(listFix)
    return listFix

def EEGByFixation(link, listFixation):
    eeg_raw = mne.io.read_raw_edf(link + "/EEG.edf")
    print(eeg_raw.info)
    eeg_data_new = eeg_raw.copy().load_data().filter(l_freq=4., h_freq=8.)
    eegTs = pd.read_csv(link + '/EEGTimeStamp.txt', names = ['TimeStamp'])
    eegTs = eegTs.sort_values(by=['TimeStamp'])
    print("Time in EEGTS by Frame number: ",len(eegTs) / 128, "*"*20)
    print("Time in EEGTS by TS: ",eegTs["TimeStamp"].iloc[-1] - eegTs["TimeStamp"].iloc[0], "*"*20)
    timeInTs = len(eegTs) / 128
    timeInFrame = eegTs["TimeStamp"].iloc[-1] - eegTs["TimeStamp"].iloc[0]
    if abs(timeInTs - timeInFrame) > 2:
        return [], []
    eegStartTs = eegTs["TimeStamp"][0]
    listEEG = []
    listDiff = []
    for idx, rangeFix in enumerate(listFixation):
        startFix, stopFix = listFixation[idx]
        print("Fix info: ",startFix, " ", stopFix, " " ,stopFix - startFix)
        if stopFix - startFix <= 0.1:
            continue

        # get EEG frame by index
        start = [eegTs[eegTs['TimeStamp'] >= startFix].index[0] / 128 + 0.01]
        # print(eegTs[eegTs['TimeStamp'] <= stopFix])
        stop = [eegTs[eegTs['TimeStamp'] <= stopFix].index[-1] / 128]
        # print(start," " ,stop, " ", stop - start )

        diffTs = (stopFix - startFix) - (stop[0] - start[0])
        # print(diffTs)
        if start[0] < stop[0]:
            try:
                tmp = eeg_data_new.copy().crop(start[0], stop[0])
                # chon 4 channels
                # Cz, Fz, Fp1, F7, F3, FC1, C3, FC5, FT9, T7, CP5, CP1, P3, P7, PO9, O1, Pz, Oz, O2, PO10, P8, P4, CP2, CP6, T8, FT10, FC6, C4, FC2, F4, F8, Fp2
                # matrix = tmp.get_data(picks = ['C3', 'Cz', 'C4', 'CP1', 'CP2']).T
                
                # uncomment neu chon toan bo channel
                tmp = tmp.to_data_frame()
                matrix = tmp.iloc[:, 1:].to_numpy()
                
                
                numFrame = matrix.shape[0]
                numFrame2time = numFrame / 128
                diffEEGvET = (stopFix - startFix) - numFrame2time
                diffEEGvTs = (stop[0] - start[0]) - numFrame2time
                # print(diffEEGvET, diffEEGvTs)
                listDiff.append([diffTs, diffEEGvET, diffEEGvTs])
                listEEG.append(matrix)
            except Exception as e:
                print("error")
    return listEEG, listDiff


def extractSub(path):
    samples = os.listdir(path)
    print(samples)
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
            listFixation = ET2Fixation(et_raw)

            scenarioNum = -1
            with open(jsonpath) as json_file:
                datajs = json.load(json_file)
                scenarioNum = getScenNumber(datajs)
            listEEG, listDiff = EEGByFixation(samplePath, listFixation)
            data.append([scenarioNum, listEEG, listFixation])
    return data


def extractData(testType = 0, windowSize = 120):
    datas = []
    list_dir = subInVIN()
    print(list_dir)
    for dir in list_dir:
        data = extractSub(dir)
        datas.append(data)

    return datas

def get_data(dataName, expType = 1, expIndex = 1):
    # get data rgd to dataName
    # type = [1, 2, 3] rgd with ["none", "scenario", "phase"]
    dataList = ["VIN", "PLOS", "Phy"]
    if dataName not in dataList:
        return None
    if dataName == "VIN":
        list_dir = subInVIN()
        print(list_dir)
        data = getDataByDir(list_dir)
        data = convertData2Numpy(data, expType, expIndex)
    elif dataName == "PLOS":
        pass
    else:
        pass

    return data


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
            mat = np.asarray(inputs)
            s1, s2, s3, s4 = mat.shape
            mat = mat.reshape(s1*s2, s3, s4)
            coMat = mat.mean(axis = 0)
            Adj = np.abs(np.corrcoef(coMat[:,:].T))
            matAdj = torch.Tensor(Adj).to(device)
            outputs = model(inputs, matAdj)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += [loss.item()]
            total_loss += loss.item()
            if (i + 1) % log_batch == 0:    # print every 200 mini-batches
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_batch))
                percent = int(i *50/ len(trainLoader))
                remain = 50 - percent
                sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#'*percent + '-'*remain, percent * 2, np.mean(running_loss)))
                sys.stdout.flush()

                #if (i + 1) / log_batch >= 10:
                #    break

        mean_loss = total_loss / len(trainLoader)
        llos.append(mean_loss)
        scheduler.step()
        sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#'*50, 100, mean_loss))
        sys.stdout.flush()
        acc = evaluateModel(model, plotConfusion = False, dataLoader = validLoader, n_class=n_class)
        accTrain = evaluateModel(model, plotConfusion = False, dataLoader = trainLoader, n_class=n_class)
    return model, llos, acc, accTrain

def vis():
    embeddings = []

    model.eval()
    for X in validLoader:
        e = model(X[0].to(device))
        embeddings.append(e.cpu().detach().numpy())

    embeddings = np.concatenate(embeddings)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    transformed = tsne.fit_transform(embeddings)

    import seaborn as sns
    palette = sns.color_palette("bright", 7)
    sns.scatterplot(
        x=transformed[:,0],
        y=transformed[:,1],
        hue=validLoader.dataset.y,
        legend='full',
        palette=palette
    )


if __name__ == "__main__":
    listPaths = []
    numberObject = 10
    counter = 0 
    for x in os.listdir("../DataVIN/Official/"):
        if x != "BN001" and x != "K317" and x!= "BN002" and x!= "K299" and x!= "K305":
            listPaths.append("Official/"+x)
            counter += 1
        if counter > numberObject:
            break
    print(listPaths)

    if not os.path.exists("./data.npy"):
        datas = extractData()
        print("Number of subjects in data: ", len(datas))
        PreProDatas = preprocessData(datas, 128)
        np.save("./data", PreProDatas)
    else:
        PreProDatas = np.load("./data.npy", allow_pickle= True)


    # analyzer general
    # analyze(PreProDatas)
    for i in range(len(PreProDatas)):
        print(listPaths[i])
        # analyzeSub(PreProDatas[i], i)

    scAcc = []
    scAccTrain = []
    for sc in range(1, 9):
    # test by scenario
        X_train, y_train, X_test, y_test = getDataScenario(PreProDatas, sc)
        X_train, y_train = augmentData(np.asarray(X_train), np.asarray(y_train), labels= [x for x in range(numberObject)])
        # analyzeTrainData(y_train, "original distribution training")
        # analyzeTrainData(y_test, "original distribution testing")
        # reshape X_train, X_test

        n_samples, n_timestamp, n_channels = X_train.shape
        X_train = X_train.reshape((n_samples, n_timestamp, n_channels, 1))
        X_train = np.transpose(X_train, (0, 3, 1, 2))

        n_samples, n_timestamp, n_channels = X_test.shape
        X_test = X_test.reshape((n_samples, n_timestamp, n_channels, 1))
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        trainLoader, validLoader = TrainTestLoader([X_train, y_train, X_test, y_test])
        num_class = len(np.unique(y_train))
        
        listModelName = []
        model = chooseModel("GCN", num_class, input_size = (1, X_train.shape[2], X_train.shape[3]))
        print("Model architecture >>>", model)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 3e-3
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 3

        _, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, n_class= num_class, log_batch=len(trainLoader) // 30)
    

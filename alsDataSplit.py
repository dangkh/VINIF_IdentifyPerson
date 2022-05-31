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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import sqrtm

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
        subPath = path + '/' + sub
        infoPath = path + '/' + sub + '/info.json'
        if os.path.isfile(infoPath) and os.path.isdir(subPath):
            list_dir.append(subPath)
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



def EEGExtractor(link):
    eeg_raw = mne.io.read_raw_edf(link + "/EEG.edf")
    print(eeg_raw.info)
    eeg_data_new = eeg_raw.copy().load_data().filter(l_freq=0.4, h_freq=50.)
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
    for annos in eeg_data_new.annotations:
        if annos["description"] != 'Thinking':
            continue
        print(annos)
        start = annos["onset"] + 0.1
        y = annos["duration"]
        stop = start + y - 0.1
        tmp = eeg_data_new.copy().crop(start, stop)
        # chon 4 channels
        # Cz, Fz, Fp1, F7, F3, FC1, C3, FC5, FT9, T7, CP5, CP1, P3, P7, PO9, O1, Pz, Oz, O2, PO10, P8, P4, CP2, CP6, T8, FT10, FC6, C4, FC2, F4, F8, Fp2
        # matrix = tmp.get_data(picks = ['C3', 'Cz', 'C4', 'CP1', 'CP2']).T

        # uncomment neu chon toan bo channel
        tmp = tmp.to_data_frame()
        matrix = tmp.iloc[:, 1:].to_numpy()
        listEEG.append(matrix)
    return listEEG, ""


def extractSub(path):
    samples = os.listdir(path)
    print(samples)
    data = []
    for idx, sample in enumerate(samples):
        samplePath = path + '/' + sample + '/'
        jsonpath = samplePath + "scenario.json"
        if os.path.isdir(samplePath):
            print(samplePath)
            scenarioNum = -1
            with open(jsonpath) as json_file:
                datajs = json.load(json_file)
                scenarioNum = getScenNumber(datajs)
            listEEG, _ = EEGExtractor(samplePath)
            data.append([scenarioNum - 1, listEEG, _])
    return data


def extractData(testType=0, windowSize=120):
    datas = []

    list_dir = []
    for sub in listPaths:
        list_dir.append(sub)

    # print(list_dir)
    for dir in list_dir:
        data = extractSub(dir)
        datas.append(data)

    return datas


def get_data(dataName, expType=1, expIndex=1):
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


def trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, n_class, log_batch, adj, testAdj):
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
            loss = criterion(outputs, labels)
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
        acc = evaluateModel(model, plotConfusion=False, dataLoader=validLoader, n_class=n_class, adj=testAdj)
        accTrain = evaluateModel(model, plotConfusion=False, dataLoader=trainLoader, n_class=n_class, adj=adj)
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
        x=transformed[:, 0],
        y=transformed[:, 1],
        hue=validLoader.dataset.y,
        legend='full',
        palette=palette
    )


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


if __name__ == "__main__":
    listPaths = []
    numberObject = 17
    counter = 0
    prePath = "D:/als-patients"
    for x in os.listdir(prePath):
        listPaths.append(prePath + '/' + x)
        counter += 1
        if counter > numberObject:
            break
    # counter = 0 
    # prePath = "D:/normal"
    # for x in os.listdir(prePath):
    #     listPaths.append(prePath + '/' + x)
    #     counter += 1
    #     if counter > numberObject:
    #         break

    if not os.path.exists("./dataCompare.npy"):
        datas = extractData()
        print("Number of subjects in data: ", len(datas))
        PreProDatas = preprocessData(datas, 128)
        np.save("./dataCompare", PreProDatas)
    else:
        PreProDatas = np.load("./dataCompare.npy", allow_pickle=True)

    # # analyzer general
    # # analyze(PreProDatas)
    # for i in range(len(PreProDatas)):
    #     print(listPaths[i])
        # analyzeSub(PreProDatas[i], i)

    # test covariance matrix
    # # test by scenario
    # X_f, y_f = getDataMI_All(PreProDatas)
    # X_f1, y_f1 = getDataMI_All_byPerson(PreProDatas, personID = 1)
    # X_f0, y_f0 = getDataMI_All_byPerson(PreProDatas, personID = 0)
    # X_f = np.vstack([X_f1, X_f0])
    # normR = getNormR(X_f)
    # normR = np.linalg.inv(normR)
    # normR = sqrtm(normR)
    # print(normR.shape)
    # np.savetxt('normR.txt', normR, fmt= '%.3f')
    # stop
    # for ij in range(1, 3):
    #     X_f, y_f = getDataMI_All_byPerson(PreProDatas, personID = ij)
    #     # normR = getNormR(X_f)
    #     # normR = np.linalg.inv(normR)
    #     # normR = sqrtm(normR)
    #     # print(normR.shape)
    #     # np.savetxt('normR.txt', normR, fmt= '%.3f')
    #     # stop
    #     normR = np.loadtxt('normR.txt')
    #     tmp = []
    #     for ii in range(len(X_f)):
    #         Xnew = np.matmul(normR, X_f[ii])
    #         tmp.append(Xnew)
    #     X_f = np.asarray(tmp)
    #     test = np.mean(X_f, axis = 0)
    #     print(test.shape)
    #     covTest = np.matmul(test.T, test)
    #     # plt.clf()
    #     f, ax = plt.subplots(figsize =(9, 8))
    #     import seaborn as sns
    #     sns.heatmap(covTest, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
    #     # print(X_f.shape)
    #     plt.savefig(str(ij) + '.png')
    # # plt.show()
    # stop
    X_f, y_f = getDataMI_All(PreProDatas)
    y_f[np.where((y_f > 1) & (y_f <= 9)) ] = 6
    y_f[np.where(y_f <= 1 ) ] = 0
    y_f[np.where(y_f == 6 ) ] = 1
    y_f[np.where(y_f == 10 ) ] = 2
    y_f[np.where((y_f > 10) & (y_f <= 13) ) ] = 3
    y_f[np.where(y_f == 14) ] = 4
    y_f[np.where(y_f > 14) ] = 5
    X_train = X_f[np.where(y_f % 2 == 0)]
    y_train = y_f[np.where(y_f % 2 == 0)]
    X_valid = X_f[np.where(y_f % 2 == 1)]
    y_valid = y_f[np.where(y_f % 2 == 1)]
    # s1, s2, s3 = X_train.shape
    matAdj = calculateAdj(np.expand_dims(X_f, 0))
    # testAdj = calculateAdj(np.expand_dims(X_test, 0))
    # X_train, y_train = augmentData(X_f, y_f,
    #                                labels=[x for x in range(1,10,1)])
    # X_train, y_train = X_f, y_f
    # # analyzeTrainData(y_train, "original distribution training")
    # # analyzeTrainData(y_test, "original distribution testing")
    # # reshape X_train, X_test
    # print(X_train.shape)
    n_samples, n_timestamp, n_channels = X_train.shape
    # print(X_train[0])
    
    # normR = getNormR(X_train)
    # normR = sqrtm(normR)
    # normR = np.linalg.inv(normR)
    # # np.savetxt('normR.txt', normR, fmt= '%.3f')
    # # stop
    # tmp = []
    # for ii in range(len(X_train)):
    #     Xnew = np.matmul(normR, X_train[ii])
    #     tmp.append(Xnew)
    # X_train = np.asarray(tmp)
    # tmp = []
    # for ii in range(len(X_valid)):
    #     Xnew = np.matmul(normR, X_valid[ii])
    #     tmp.append(Xnew)
    # X_valid = np.asarray(tmp)
    # print(X_train[0])
    # stop
       

    X_train = X_train.reshape((n_samples, n_timestamp, n_channels, 1))
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    trainLoader, validLoader = TrainTestLoader([X_train, y_train])
    if not os.path.exists("./alsModel.pt")     :
        

        # n_samples, n_timestamp, n_channels = X_test.shape
        # X_test = X_test.reshape((n_samples, n_timestamp, n_channels, 1))
        # X_test = np.transpose(X_test, (0, 3, 1, 2))
        
        num_class = len(np.unique(y_f))
        # listModelName = []
        model = chooseModel("CNN_LSTM", num_class, input_size=(1, X_train.shape[2], X_train.shape[3]))
        print("Model architecture >>>", model)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = 10

        _, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader,
                                            validLoader, n_class=num_class, log_batch=len(trainLoader) // 24,
                                            adj=matAdj, testAdj=matAdj)
        print(acc, accTrain)
        torch.save(model.state_dict(), "alsModel.pt")
        # scAcc.append(acc)
        # scAccTrain.append(accTrain)
    model = chooseModel("CNN_LSTM", len(np.unique(y_f)), input_size=(1, X_train.shape[2], X_train.shape[3]))
    print("Model architecture >>>", model)
    model.to(device)
    model.load_state_dict(torch.load("./alsModel.pt"))
    model.fc2.register_forward_hook(get_activation('fc2'))

    
    n_samples, n_timestamp, n_channels = X_valid.shape
    tmpData = np.copy(X_valid).reshape(n_samples, n_timestamp * n_channels)
    print(tmpData.shape)

    X_f = X_valid.reshape((n_samples, n_timestamp, n_channels, 1))

    X_f = np.transpose(X_f, (0, 3, 1, 2))
    vis_dataset = EEG_data(X_f, y_valid)
    vis_loader = torch.utils.data.DataLoader(dataset=vis_dataset,
        batch_size=32, shuffle=False)
    list_activation = []
    colors = ['r', 'b', 'y', 'g', 'plum', 'purple']
    with torch.no_grad():
        for i, data in enumerate(vis_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            list_activation.extend(activation['fc2'].numpy())
    # pca = PCA(n_components = 3)
    list_activation = np.asarray(list_activation)
    # newactivation = pca.fit(list_activation).transform(list_activation)
    # newactivation = np.asarray(list_activation)
    fig = plt.figure(1, figsize = (10, 10))
    # plt.clf()
    ax = Axes3D(fig, rect = [0, 0, 0.95, 1], elev = 48, azim = 134)
    plt.cla()
    # numData = 2683
    # ax.scatter(newactivation[:(numData//2),0], newactivation[:(numData//2), 1], c='b', s=0.3)
    # ax.scatter(newactivation[int(numData*1.5):,0], newactivation[int(numData*1.5):, 1], c= 'r')
    # plt.show()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(0.5, 0.5)
    paint = [colors[x] for x in y_valid]
    tmp = np.unique(y_valid)
    print(tmp)
    print([colors[x] for x in tmp])
    print(len(y_valid[np.where(y_valid == 0)]))
    ax.scatter(list_activation[:,0], list_activation[:, 1], list_activation[:, 2], c= paint)
    plt.show()

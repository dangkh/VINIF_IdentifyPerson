from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nets import *


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
        mean = np.mean(datas, axis=3, keepdims=True)
        std = np.std(datas, axis=3, keepdims=True)
        self.X = (datas - mean) / std
        self.X = self.X.astype(np.float32) * 1e3

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
        dataRemove, targetRemove = randomRemoveSample(X_source, y_source)
        dataSwap, targetSwap = randomSwapSample(X_source, y_source)
        newXs.extend(datanoise)
        newXs.extend(dataRemove)
        newXs.extend(dataSwap)
        newYs.extend(targetnoise)
        newYs.extend(targetRemove)
        newYs.extend(targetSwap)
    newXs.extend(Xs)
    newYs.extend(Ys)
    return np.asarray(newXs), np.asarray(newYs)

def chooseModel(modelName, num_class, input_size = None):
    if modelName == "CNN2D":
        keys = list(paramsCNN2D)
        d = {
            'kernel_size': paramsCNN2D['kernel_size'][1],
            'conv_channels': paramsCNN2D['conv_channels'][1]
        }
        model = CNN2D(input_size    = input_size,
                        kernel_size   = d['kernel_size'],
                        conv_channels = d['conv_channels'],
                        dense_size    = 128,
                        dropout       = 0.5, 
                        nclass = num_class)
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
                        dropout       = 0.5, 
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


def evaluateModel(model, plotConfusion, dataLoader, n_class, adj):
    counter = 0
    total = 0
    preds = []
    trueLabel = []
    model.eval()
    for idx, data in enumerate(dataLoader):
        xx, yy = data
        trueLabel.extend(yy.numpy())
        total += len(yy)
        xx = xx.to(device)
        with torch.no_grad():
            pred = model(xx, adj)
            # pred = model(xx)
            res = torch.argmax(pred, 1)
            if torch.cuda.is_available():
                res = res.cpu()
            preds.extend(res.numpy())
            for id, ypred in enumerate(res):
                if ypred == yy[id].item():
                    counter += 1
    print('acc: {:1f}%'.format(100 * counter / total))
    if plotConfusion:
        plotCl = [str(x) for x in range(n_class)]
        plot_confusion_matrix(trueLabel, preds, classes= plotCl, normalize=True, title='Validation confusion matrix')


# def trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, n_class, log_batch):
#     llos = []
#     best_acc = 0
#     for epoch in range(n_epochs):  # loop over the dataset multiple times
#         model.train()
#         print("")
#         print("epoch:  {0} / {1}   ".format(epoch, n_epochs))
#         running_loss = []
#         total_loss = 0
#         for i, data in enumerate(trainLoader):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data
#             labels = labels.type(torch.LongTensor)
#             # CUDA
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += [loss.item()]
#             total_loss += loss.item()
#             if (i + 1) % log_batch == 0:    # print every 200 mini-batches
#                 # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_batch))
#                 percent = int(i *50/ len(trainLoader))
#                 remain = 50 - percent
#                 sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#'*percent + '-'*remain, percent * 2, np.mean(running_loss)))
#                 sys.stdout.flush()

#                 #if (i + 1) / log_batch >= 10:
#                 #    break

#         mean_loss = total_loss / len(trainLoader)
#         llos.append(mean_loss)
#         scheduler.step()
#         sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#'*50, 100, mean_loss))
#         sys.stdout.flush()
#         acc = evaluateModel(model, plotConfusion = False, dataLoader = validLoader, n_class=n_class)
#     return model, llos    

def calculateAdj(inputMat):
    mat = np.asarray(inputMat)
    s1, s2, s3, s4 = mat.shape
    mat = mat.reshape(s1 * s2, s3, s4)
    coMat = mat.mean(axis=0)
    Adj = np.corrcoef(coMat[:, :].T)
    # print(Adj)
    D = np.array(np.sum(Adj, axis=0))
    D = np.sqrt(D)
    # print(np.matrix(np.diag(D)).shape)
    # print(np.diag(D).shape)
    # print(np.diag(D))
    # print(np.matrix(np.diag(D)))

    # print(np.diag(D).shape)
    # print(np.matrix(np.diag(D)).shape)

    D = np.diag(D)
    matAdj = torch.Tensor(np.linalg.inv(D) * Adj * np.linalg.inv(D)).to(device)
    # matAdj = get_adj(Adj)
    return matAdj

def get_adj(adj):
    """
    build symmetric adjacency matrix
    @param adj:
    @return:
    """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
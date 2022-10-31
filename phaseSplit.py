# import numpy as np
# import matplotlib.pyplot as plt

# import mne
# from mne import io
# from mne.datasets import sample
# from scipy.spatial import distance as dist
# import matplotlib.pyplot as plt
# import time
# import os
# import json
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# import torch
# import sys
# from ultis import *
# from nets import *

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# listPaths = []
# counter = 0 
# for x in os.listdir("/content/drive/MyDrive/DataVIN/Official/"):
#     if x != "BN001" and x != "K317" and x!= "BN002" and x!= "K299" and x!= "K305":
#         listPaths.append("Official/"+x)
#         counter += 1
#     if counter > 10:
#         break
# print(listPaths)

# def checkSubFolder(path):
#     # return list folder contain data according to input path
#     list_dir = []
#     subs = os.listdir(path)
#     for sub in subs:
#         subPath = path + '/'+ sub
#         infoPath = path + '/' + sub + '/info.json'
#         if os.path.isfile(infoPath) and os.path.isdir(subPath):
#             list_dir.append(subPath)
#     return list_dir

# def subInVIN():
#     # return list folder contain data in VIN project
#     list_dir = []
#     list_dir.extend(checkSubFolder("/mnt/hdd/VINIF/DataVIN"))
#     list_dir.extend(checkSubFolder("/mnt/hdd/VINIF/DataVIN/Official"))
#     return list_dir

# def getScenNumber(jsfile):
#     # return scenario Number rgds to input jsfile
#     res = -1
#     listkey = ["scenarioId", "RecPlanEdit", "scenarioNumber"]
#     for key in listkey:
#         try:
#             res = jsfile[key]
#         except Exception as e:
#             print('error key' + key)
#     return res

# def getDataByDir(listSub):
#     # return data with given listSub in EDF format
#     data = []
#     for idx, subject in enumerate(listSub):
#         samples = os.listdir(subject)
#         for sample in samples:
#             samplePath = subject + '/' + sample + '/'
#             jsonpath = samplePath + "scenario.json"
    
#             if os.path.isdir(samplePath):
#                 scenarioNum = -1
#                 with open(jsonpath) as json_file:
#                     datajs = json.load(json_file)
#                     scenarioNum = getScenNumber(datajs)
#                 eeg_raw = mne.io.read_raw_edf(samplePath + "/EEG.edf")
#                 eeg_data_new = eeg_raw.copy().load_data()
#                 data.append([idx, eeg_data_new, scenarioNum])
#     return data

# def convertData2Numpy(data, expType, expIndex, columnLoop = True):
#     newdata = []
#     for idx, eeg, scenraioNum in data:
#         if idx > 2 and idx != 10:
#             annos = eeg.annotations
#             counter = 0
#             if len(annos) > 0:
#                 for idy in range(len(annos)):
#                     if annos[idy]['description'] == "Thinking":
#                         test = 0
#                         if expType == 3:
#                             counter += 1
#                             if counter == expIndex:
#                                 test = 1
#                         elif expType == 2:
#                             if scenarioNum == expIndex:
#                                 test = 1
#                         onset = annos[idy]['onset']
#                         duration = annos[idy]['duration']
#                         # print(onset, duration)
#                         tmp = eeg.copy().crop(onset, onset + duration)
#                         tmp = tmp.to_data_frame()
#                         matrix = tmp.iloc[:, 1:].to_numpy()
#                         if columnLoop:
#                             endMatrix = np.copy(matrix[:, 27:32])
#                             newmatrix = np.hstack([endMatrix, matrix])
#                             newdata.append([idx, newmatrix, test])
#                         else:
#                             newdata.append([idx, matrix, test])
#     return newdata


# def get_data(dataName, expType = 1, expIndex = 1):
#     # get data rgd to dataName
#     # type = [1, 2, 3] rgd with ["none", "scenario", "phase"]
#     dataList = ["VIN", "PLOS", "Phy"]
#     if dataName not in dataList:
#         return None
#     if dataName == "VIN":
#         list_dir = subInVIN()
#         print(list_dir)
#         data = getDataByDir(list_dir)
#         data = convertData2Numpy(data, expType, expIndex)
#     elif dataName == "PLOS":
#         pass
#     else:
#         pass

#     return data

# def chooseModel(modelName, num_class, input_size = None):
#     if modelName == "CNN2D":
#         keys = list(paramsCNN2D)
#         input_size = datas[0].shape
#         d = {
#             'kernel_size': paramsCNN2D['kernel_size'][1],
#             'conv_channels': paramsCNN2D['conv_channels'][1]
#         }
#         model = CNN2D(input_size    = input_size,
#                         kernel_size   = d['kernel_size'],
#                         conv_channels = d['conv_channels'],
#                         dense_size    = 128,
#                         dropout       = 0.5, 
#                         nclass = num_class)
#     elif modelName == "CNN_LSTM":
#         keys = list(paramsCNN2D)
#         input_size = datas[0].shape
#         d = {
#             'kernel_size': paramsCNN2D['kernel_size'][1],
#             'conv_channels': paramsCNN2D['conv_channels'][1]
#         }
#         model = CNN_LSTM(input_size    = input_size,
#                         kernel_size   = d['kernel_size'],
#                         conv_channels = d['conv_channels'],
#                         dense_size    = 128,
#                         dropout       = 0.5, 
#                         nclass = num_class)
#     else:
#         model = WvConvNet(num_class, 6, 2, drop_rate=0.5, flatten=True, input_size = input_size)
#         print(model)
#     return model


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
#         accTrain = evaluateModel(model, plotConfusion = False, dataLoader = trainLoader, n_class=n_class)
#     return model, llos, acc, accTrain

# def vis():
#     embeddings = []

#     model.eval()
#     for X in validLoader:
#         e = model(X[0].to(device))
#         embeddings.append(e.cpu().detach().numpy())

#     embeddings = np.concatenate(embeddings)

#     from sklearn.manifold import TSNE
#     tsne = TSNE(n_components=2)
#     transformed = tsne.fit_transform(embeddings)

#     import seaborn as sns
#     palette = sns.color_palette("bright", 7)
#     sns.scatterplot(
#         x=transformed[:,0],
#         y=transformed[:,1],
#         hue=validLoader.dataset.y,
#         legend='full',
#         palette=palette
#     )


# if __name__ == "__main__":
#     phaseAcc = []
#     phaseAccTrain = []
#     for phase in range(3):
#         data = get_data("VIN", expType = 3, expIndex = phase+1)
#         datas, targets, data_indexes = preprocessData(data, augment = True)
#         datas.shape
#         print(datas.shape)
#         n_samples, n_channels, n_timestamp = datas.shape
#         datas = datas.reshape((n_samples, n_channels, n_timestamp, 1))
#         print(datas.shape)
#         datas = np.transpose(datas, (0, 3, 1, 2))
#         print(datas.shape)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         num_class = len(np.unique(targets))
#         X_train, y_train, X_test, y_test = trainTestSplit(datas, targets, data_indexes)
#         trainLoader, validLoader = TrainTestLoader([X_train, y_train, X_test, y_test])
#         listModelName = []
#         model = chooseModel("CNN2D", num_class, input_size = datas[0].shape)
#         print("Model architecture >>>", model)
#         model.to(device)
#         criterion = nn.CrossEntropyLoss()
#         lr = 1e-4
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
#         n_epochs = 50

#         model, llos, acc, accTrain = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, n_class= num_class, log_batch=len(trainLoader) / 10)

#         torch.save(model.state_dict(), "./model_Identify_triplet.pt")

#         vis()
#         plt.show()
#         phaseAcc.append(acc)
#         phaseAccTrain.append(accTrain)
#     print(np.asarray(phaseAcc).mean())
#     print(np.asarray(phaseAccTrain).mean())
    

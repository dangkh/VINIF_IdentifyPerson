from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

class EEG_data(Dataset):
    """EEG_data set"""
    
    def __init__(self, datas, targets, transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]), train = True):
        self.y = targets
        self.X = datas
        self.transform = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])

colors = ['red', 'green', 'blue', 'black',
              'yellow', 'pink', 'orange', 'brown',
              'purple', 'charcoal', 'navy', 'emerald', 'turquoise']

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    tmp = np.unique(targets)
    counter = 0
    for idx in tmp:
        inds = np.where(targets==idx)[0]
        for id in inds:
            plt.scatter(embeddings[id][0][1], embeddings[id][1][2], alpha=0.5, color=colors[counter])
        counter += 1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(tmp)

def checkSubFolder(path):
    subs = os.listdir(path)
    for sub in subs:
        subPath = path + '/'+ sub
        infoPath = path + '/' + sub + '/info.json'
        if os.path.isfile(infoPath) and os.path.isdir(subPath):
            list_subject.append(subPath)


def chunk(matrix, step_size = 128, window_size = 128):
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


def chunk_matrix(list_data, list_target, step_size = 32, window_size = 128):
    list_matries = []
    list_ys = []
    for idx, matrix in enumerate(list_data):
        matries = chunk(matrix)
        list_matries.extend(matries)
        y_matries = [list_target[idx].astype(int)] * len(matries)
        list_ys.extend(y_matries)

  return list_matries, list_ys

def addNoise(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        tmpTarget = [0]*12
        matrix = np.copy(data[idx])
        noise = np.random.normal(0, 0.05, size= matrix.shape)
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
        numRandom = 15
        listFrame = []
        for id in range(numRandom):
          tmp = np.random.randint(1,128)
          listFrame.append(tmp)
        for f in listFrame:
          if f > 1 and f < 126:
            matrix[f] = matrix[f-1] + matrix[f+1] / 2
        # print(matrix.shape)
        list_newdata.append(matrix)
        list_newtarget.append(target[idx])
  return list_newdata, list_newtarget

def randomSwapSample(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        matrix = np.copy(data[idx])
        numRandom = 8
        listFrame = []
        for id in range(numRandom):
            tmp = np.random.randint(1,128)
            listFrame.append(tmp)
        listFrame = np.sort(listFrame)
        for idy, v in  enumerate(listFrame):
            if idy > 0 and listFrame[idy] < listFrame[idy-1]:
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
        listFrame = np.append(listFrame ,127)
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
        # print(finalMatrix.shape)
        # if finalMatrix.shape[0] == 128:
        #   print(listFrame)
        list_newdata.append(finalMatrix)
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget
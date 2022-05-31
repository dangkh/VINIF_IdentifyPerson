import numpy as np
import argparse
import os
from ultis import *



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'train')
    parser.add_argument('--input', help = 'input data dir')
    parser.add_argument('--modelName', help = 'name of model')
    parser.add_argument('--bandL', help = 'band filter', default = 4.0)
    parser.add_argument('--bandR', help = 'band filter', default = 50.0)
    parser.add_argument('--eaNorm', help = 'EA norm')
    parser.add_argument('--windowSize', help = 'windowSize', default = 120)
    parser.add_argument('--trainTestSeperate', help = 'train first then test. if not, train and test are splitted randomly', default = False)
    parser.add_argument('--trainTestSession', help = 'train test are splitted by session', default = True)
    args = parser.parse_args()
    print(args)


    listPaths = []
    numberObject = 17
    counter = 0

    prePath = args.input
    for x in os.listdir(prePath):
        listPaths.append(prePath + '/' + x)
        counter += 1
        if counter > numberObject:
            break

    tmp = 'trainTestRandom'
    if args.trainTestSeperate:
        tmp = 'trainTestSeperate'
    if args.trainTestSession and args.trainTestSeperate:
        tmp = 'trainTestSession'

    dataLink = prePath + '/' + 'band_' + str(args.bandL) + '_' + str(args.bandR) + '_EA_' + str(args.eaNorm) + tmp + '.npy'
    print(dataLink)
    # stop
    if not os.path.exists(dataLink):
        info = {'bandL': args.bandL, 'bandR': args.bandR, 'windowSize': args.windowSize, 'listPaths': listPaths}
        datas = extractData_byInfo(info)
        print("Number of subjects in data: ", len(datas))
        PreProDatas = preprocessData(datas, 128)
        np.save(dataLink, PreProDatas)
    else:
        PreProDatas = np.load(dataLink, allow_pickle=True)

    # # analyzer general
    # # analyze(PreProDatas)
    # for i in range(len(PreProDatas)):
    #     print(listPaths[i])
        # analyzeSub(PreProDatas[i], i)

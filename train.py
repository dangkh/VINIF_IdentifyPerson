import numpy as np
import argparse
import os
from ultis import *

channelCombos = [   
                    ['C3', 'Cz', 'C4', 'CP1', 'CP2'], ['F3', 'F4', 'C3', 'C4'], ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8'], 
                    ['Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 
                    'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2']
                ]
persons = [10, 9, 6]
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
    parser.add_argument('--channelType', help = 'channel seclection in : {}'.format(channelCombos), default = 3)
    parser.add_argument('--windowSize', help = 'windowSize', default = 120)
    parser.add_argument('--extractType', help = 'type of extraction in eeg. Fixation: True. All: False', default = True)
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

    tmpExtract = 'Fixation'
    if not args.extractType:
        tmpExtract = 'All'
    tmp = 'trainTestRandom'
    if args.trainTestSeperate:
        tmp = 'trainTestSeperate'
    if args.trainTestSession and args.trainTestSeperate:
        tmp = 'trainTestSession'

    dataLink = prePath + '/' + 'band_' + str(args.bandL) + '_' + str(args.bandR) + '_channelType_'+ str(args.channelType) + '_' + tmp + '_' + tmpExtract+'.npy'
    print(dataLink)
    if not os.path.exists(dataLink):
        info = {
            'bandL':        args.bandL, 
            'bandR':        args.bandR, 
            'windowSize':   args.windowSize, 
            'listPaths':    listPaths,
            'EA':           args.eaNorm,
            'extractType':  args.extractType,
            'channelType':  args.channelType
            }
        datas = extractData_byInfo(info)
        print("Number of subjects in data: ", len(datas))
        PreProDatas = preprocessData(datas, 128)
        np.save(dataLink, PreProDatas)
    else:
        PreProDatas = np.load(dataLink, allow_pickle=True)

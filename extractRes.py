import os
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extraction Result')
    parser.add_argument('--inputDir', help='input directory')
    parser.add_argument('--outputFile', help='file name contain extracted result', default='./result.txt')
    listRes = [[], [], []]
    args = parser.parse_args()
    print(args)
    f = open(args.inputDir, "r")
    for line in f:
        if line[:6] == 'Result':
            infos = line.split(' ')
            acc, std = infos[-2], infos[-1][:3]
            accs = acc.split('.')
            acc = accs[0]+'.'+accs[1][:2]
            if types[1:3] == 'EA':
                listRes[0].append([acc, std])
            elif types[1:4] == 'DEA':
                listRes[1].append([acc, std])
            else:
                listRes[2].append([acc, std])
        elif line[:4] == 'Name':
            infos = line.split(',')
            types = infos[3].split('=')[-1]
            print(infos[0], infos[1])
# print(listRes)
for tt in range(3):
    line = ""
    for ii in range(len(listRes[tt])):
        if len(line) > 0:
            line += ' & '
        line = line + listRes[tt][ii][0] + " $\pm$ " + listRes[tt][ii][1]
    print(line)
import numpy as np

import mne
import matplotlib.pyplot as plt
import time

import os 
list_subject = []

def checkSubFolder(path):
	subs = os.listdir(path)
	for sub in subs:
		subPath = path + '/'+ sub
		infoPath = path + '/' + sub + '/info.json'
		if os.path.isfile(infoPath) and os.path.isdir(subPath):
			list_subject.append(subPath)

if __name__ == "__main__":
	checkSubFolder("/mnt/hdd/VINIF/DataVIN")
	checkSubFolder("/mnt/hdd/VINIF/DataVIN/Official")
	print(list_subject)

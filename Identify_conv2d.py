import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import cv2
import imutils
from imutils import face_utils
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
	checkSubFolder("/content/drive/MyDrive/DataVIN")
	checkSubFolder("/content/drive/MyDrive/DataVIN/Official")
	print(list_subject)
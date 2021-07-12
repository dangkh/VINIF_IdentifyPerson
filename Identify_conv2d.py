import numpy as np

import mne
import matplotlib.pyplot as plt
import time

import os

from ultis import *
from sklearn.model_selection import train_test_split
from nets import *

list_subject = []

if __name__ == "__main__":
	checkSubFolder("/mnt/hdd/VINIF/DataVIN")
	checkSubFolder("/mnt/hdd/VINIF/DataVIN/Official")
	print("list subject: ", list_subject)
	data = []
	target_subject = 0
	for idx, subject in enumerate(list_subject):
		samples = os.listdir(subject)
		for sample in samples:
			samplePath = subject + '/' + sample + '/'
			if os.path.isdir(samplePath):
				tmp = sample[-1:]
				eeg_raw = mne.io.read_raw_edf(samplePath + "/EEG.edf")
				eeg_data_new = eeg_raw.copy().load_data()
				data.append([idx, eeg_data_new])
	print("read data")	
	idEpoch = []
	for idx, eeg in data:
	list_thinking = []
	annos = eeg.annotations
	if len(annos) > 0:
		for idy in range(len(annos)):
			if annos[idy]['description'] == "Thinking":
				onset = annos[idy]['onset']
				duration = annos[idy]['duration']
				# print(onset, duration)
				tmp = eeg.copy().crop(onset, onset + duration)
				tmp = tmp.to_data_frame()
				matrix = tmp.iloc[:, 1:].to_numpy()
				endMatrix = np.copy(matrix[:, 27:32])
				newmatrix = np.hstack([endMatrix, matrix])
				idEpoch.append([idx, newmatrix])

	targets = []
	datas = []
	for idx, _ in enumerate(idEpoch):
		info = idEpoch[idx]
		targets.append(info[0])
		datas.append(info[1])
	targets = np.asarray(targets)
	print("unique targets: ", np.unique(targets))

	print("len data:")
	print(len(targets))

	newdata, newtarget = chunk_matrix(datas, targets)
	datanoise, targetnoise = addNoise(newdata, newtarget)
	dataRemove, targetRemove = randomRemoveSample(newdata, newtarget)
	dataSwap, targetSwap = randomSwapSample(newdata, newtarget)

	Xs = []
	Xs.extend(newdata)
	Xs.extend(datanoise)
	Xs.extend(dataRemove)
	Xs.extend(dataSwap)
	Ys = []
	Ys.extend(newtarget)
	Ys.extend(targetnoise)
	Ys.extend(targetRemove)
	Ys.extend(targetSwap)
	print(len(Xs), len(Ys))

	(unique, counts) = np.unique(np.asarray(Ys), return_counts=True)
	frequencies = np.asarray((unique, counts)).T
	print(frequencies)

	print(unique)
	print(counts)
	plt.bar(unique, counts)
	plt.show()


	X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.1, random_state=42)
	batch_size = 32

	train_dataset = EEG_data(X_train, y_train)
	test_dataset = EEG_data(X_test, y_test)


	train_loader_simple = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader_simple = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= batch_size)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	print(torch.device == "cpu")

	import torch.optim as optim
	from torch.optim import lr_scheduler
	model = LSTMNet(n_classes=15)
	model.double()
	if torch.cuda.is_available():
		model.cuda()
	criterion = nn.CrossEntropyLoss()
	lr = 1e-4
	optimizer = optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.2, last_epoch=-1)
	n_epochs = 100
	log_interval = 50
	llos = []
	lacc = []

	for epoch in range(n_epochs):  # loop over the dataset multiple times
	model.train()
	print("epoch:     ", epoch)
	running_loss = 0.0
	total_loss = 0
	for i, data in enumerate(train_loader_simple):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data
		# CUDA
		if torch.cuda.is_available():
		  inputs = inputs.cuda()
		  labels = labels.cuda()
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		total_loss += loss.item()
		if i % 50 == 49:    # print every 200 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
			running_loss = 0.0
		mean_loss = total_loss / len(train_loader_simple)
		llos.append(mean_loss)
		counter = 0
		for idx, data in enumerate(train_loader_simple):
			xx, yy = data
			if torch.cuda.is_available():
				xx = xx.cuda()
				yy = yy.cuda()
			with torch.no_grad():
				model.eval()
				pred = model(xx)
				res = torch.argmax(pred, 1)
				for i, ypred in  enumerate(res):
					if ypred == yy[i].item():
						counter += 1
		acc = counter / len(X_train)
		lacc.append(acc)
	print('Finished Training')

	counter = 0
	total = 0
	for idx, data in enumerate(test_loader_simple):
	xx, yy = data
	total += len(yy)
	# cuda
	if torch.cuda.is_available():
		xx = xx.cuda()
	with torch.no_grad():
		model.eval()
		pred = model(xx)
		res = torch.argmax(pred, 1)
		for id, ypred in enumerate(res):
			if ypred == yy[id].item():
				counter += 1
	# print(counter / total, counter, total)    
	print('acc: {:1f}%'.format(100 * counter / total))
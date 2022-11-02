import os 
import numpy as np 
import matplotlib.pyplot as plt
import random

numSubs = [x * 10 + 10 for x in range(10)]
#  HMI 
# numChannels = [1, 4, 8, 16, 32]
#  Phi
# numChannels = [1, 4, 8, 16, 32, 64]

# res = [[random.randint(90, 100) for x in range(6)] for numMethods in range(9)]
# print(res)
# fig, ax = plt.subplots()
# fig.suptitle('Accuracy with different number of channel', fontsize=14)
# colors = ['green', 'orange', 'blue']
# labels = ['SVM + PSD + Z-score', 'SVM + PSD + EA', 'SVM + PSD + DEA', 'SVM + APF + Z-score', 'SVM + APF + EA', 'SVM + APF + DEA',
# 'CNN_LSTM + RAW + Z-score', 'CNN_LSTM + RAW + EA', 'CNN_LSTM + RAW + DEA']
# for ii in range(3):
# 	ax.plot(numChannels, res[ii*3], 'o', linestyle='dotted', color = colors[ii], label = labels[ii*3])
# 	ax.plot(numChannels, res[ii*3+1], '*', linestyle='dashed', color = colors[ii], label = labels[ii*3+1])
# 	ax.plot(numChannels, res[ii*3+2], '.', linestyle='solid', color = colors[ii], label = labels[ii*3+2])
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# plt.legend(loc = 'center', framealpha = 0.2, bbox_to_anchor=(1.2, 0.5))
# ax.figure.set_size_inches(10, 5)
# ax.set_xlabel(' Number of channel ')
# ax.set_ylabel(' Accuracy (%) ')
# plt.xticks(numChannels)
# # plt.show()
# plt.savefig("Phi.png")


res = [
[  99.94   , 99.91   , 99.92   , 99.89   , 99.44   , 99.34   , 99.25   , 99.30   , 99.30 , 99.30  ], 
[  99.98   , 99.89   , 99.87   , 99.87   , 99.50   , 99.35   , 99.24   , 99.31   , 99.19 , 99.30  ],
[  96.37   , 96.79   , 94.37   , 92.95   , 91.76   , 90.74   , 89.26   , 87.74   , 87.89 , 86.5  ]]

# print(res)
fig, ax = plt.subplots()
fig.suptitle('Accuracy with different number of subject', fontsize=14)
colors = ['green', 'orange', 'blue']
labels = ['SVM + PSD + Z-score', 'SVM + PSD + EA', 'SVM + PSD + DEA', 'SVM + APF + Z-score', 'SVM + APF + EA', 'SVM + APF + DEA',
'CNN_LSTM + RAW + Z-score', 'CNN_LSTM + RAW + EA', 'CNN_LSTM + RAW + DEA']
for ii in range(1):
	ax.plot(numSubs, res[ii*3], 'o', linestyle='dotted', color = colors[ii], label = labels[ii*3])
	ax.plot(numSubs, res[ii*3+1], '*', linestyle='dashed', color = colors[ii+1], label = labels[ii*3+1])
	ax.plot(numSubs, res[ii*3+2], '.', linestyle='solid', color = colors[ii+2], label = labels[ii*3+2])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc = 'center', framealpha = 0.2, bbox_to_anchor=(1.2, 0.5))
ax.figure.set_size_inches(10, 5)
ax.set_xlabel(' Number of subject ')
ax.set_ylabel(' Accuracy (%) ')
ax.set_ylim([70, 100])
plt.xticks(numSubs)
# plt.show()
plt.savefig("phi_subject.png")

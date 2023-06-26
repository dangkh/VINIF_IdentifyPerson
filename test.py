import os 
import numpy as np 
import matplotlib.pyplot as plt
import random

numSubs = np.arange(6)*20


resAPF = np.asarray([
[64.17	,82.91	,53.23	,76.45	,50.91	,56.31], 
[94.22	,99.25  ,93.32	,96.48	,90.48	,93.21]])

ch_names = ['1', '2', '4', '8', '16', '32']
# print(res)
fig, ax = plt.subplots()
fig.suptitle('Accuracy on  HMI dataset with different channel set', fontsize=11)
colors = ['limegreen', 'moccasin', 'thistle']
width = 5.1
labels = ['AEF + Z-score', 'AEF + DDR', 'PSD + Z-score', 'PSD + DDR', 'CNN_LSTM + Z-score', 'CNN_LSTM + DDR']
rects1 = ax.bar(numSubs - width, resAPF[0], width/2, color = '#4472c4', label = labels[0])
rects2 = ax.bar(numSubs - (width/2), resAPF[1], width/2, color = '#F5B29C', label = labels[1])
# rects3 = ax.bar(numSubs, resPSD[0], width/2,  color = '#A39CF5', label = labels[2])
# rects4 = ax.bar(numSubs + (width / 2), resPSD[1], width/2,  color = '#f8c002', label = labels[3])
# rects5 = ax.bar(numSubs + width, resCNN[0], width/2, color = '#70ad47', label = labels[4])
# rects6 = ax.bar(numSubs + (width * 1.5), resCNN[1], width/2, color = '#c00000', label = labels[5])

# ax.bar_label(rects1, padding=5)
# ax.bar_label(rects2, padding=5)
# ax.bar_label(rects3, padding=5)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc = 'center', framealpha = 0.2, bbox_to_anchor=(1.2, 0.5))
ax.figure.set_size_inches(10, 5)
ax.set_ylabel(' Average Accuracy (%) ')
ax.set_xlabel(' Size of sliding window')
ax.set_xticks(numSubs, labels=ch_names)
# ax.set_xlim([70, 105])
# ax.set_ylim([0, 105])
# plt.yticks(numSubs)
# plt.show()
plt.savefig("slidingW.eps", format='eps')

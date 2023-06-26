import os 
import numpy as np 
import matplotlib.pyplot as plt
import random

numSubs = np.arange(6)*20

patterns = ["x", "|", "//" ]




# resAPF = np.asarray([
# [93.83	,91.66	,88.89	,86.51	,84.65	,84.07],
# [99.07	,99.17	,98.93	,98.48	,97.85	,97.46]
# 	])

# resPSD = np.asarray([
# [58.45, 76.28, 59.19, 77.03, 62.36, 63.05], 
# [78.45, 95.95, 90.92, 97.26, 88.90, 84.58]])

# resCNN = np.asarray([
# [98.50	,99.11	,99.04	,98.78	,98.83	,98.44],
# [99.57	,99.11	,99.80	,99.55	,99.47	,99.24]])



resAPF = np.asarray([
[64.17	,82.91	,53.23	,76.45	,50.91	,56.31], 
[94.22	,99.25  ,93.32	,96.48	,90.48	,93.21]])

resPSD = np.asarray([
[58.45	,76.28	,59.19	,77.03	,62.36	,63.05], 
[78.45	,95.95	,90.92	,97.26	,88.90	,84.58]])

resCNN = np.asarray([
[97.42	,97.80	,92.95	,97.23	,87.09	,90.24], 
[97.51	,99.21	,97.75	,98.87	,96.67	,97.03]])

ch_names = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
# print(res)
fig, ax = plt.subplots()
fig.suptitle('Accuracy on  HMI dataset with different channel set', fontsize=11)
colors = ['limegreen', 'moccasin', 'thistle']
width = 5.1
labels = ['AEF + Z-score', 'AEF + DDR', 'PSD + Z-score', 'PSD + DDR', 'CNN_LSTM + Z-score', 'CNN_LSTM + DDR']
rects1 = ax.bar(numSubs - width, resAPF[0], width/2, color = '#4472c4', label = labels[0])
rects2 = ax.bar(numSubs - (width/2), resAPF[1], width/2, color = '#F5B29C', label = labels[1])
rects3 = ax.bar(numSubs, resPSD[0], width/2,  color = '#A39CF5', label = labels[2])
rects4 = ax.bar(numSubs + (width / 2), resPSD[1], width/2,  color = '#f8c002', label = labels[3])
rects5 = ax.bar(numSubs + width, resCNN[0], width/2, color = '#70ad47', label = labels[4])
rects6 = ax.bar(numSubs + (width * 1.5), resCNN[1], width/2, color = '#c00000', label = labels[5])

# ax.bar_label(rects1, padding=5)
# ax.bar_label(rects2, padding=5)
# ax.bar_label(rects3, padding=5)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc = 'center', framealpha = 0.2, bbox_to_anchor=(1.2, 0.5))
ax.figure.set_size_inches(10, 5)
ax.set_ylabel(' Average Accuracy (%) ')
ax.set_xlabel(' Channel set')
ax.set_xticks(numSubs, labels=ch_names)
# ax.set_xlim([70, 105])
# ax.set_ylim([0, 105])
# plt.yticks(numSubs)
# plt.show()
plt.savefig("hmi_channel.eps", format='eps')

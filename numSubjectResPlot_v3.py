import os 
import numpy as np 
import matplotlib.pyplot as plt
import random

numSubs = np.asarray([16, 32, 48, 64])

patterns = ["x", "|", "//" ]

res = [
[94.06 ,   98.52 ,   98.99, 99.50] , 
[94.01 ,   98.48 ,   98.99, 99.44] , 
[44.32 ,   72.44 ,   88.38, 91.76]]

# print(res)
fig, ax = plt.subplots()
fig.suptitle('Accuracy with different number of subject', fontsize=11)
colors = ['limegreen', 'moccasin', 'thistle']
width = 3.1
labels = ['SVM + PSD + Z-score', 'SVM + PSD + EA', 'SVM + PSD + DEA', 'SVM + APF + Z-score', 'SVM + APF + EA', 'SVM + APF + DEA',
'CNN_LSTM + RAW + Z-score', 'CNN_LSTM + RAW + EA', 'CNN_LSTM + RAW + DEA']
rects1 = ax.barh(numSubs - width, res[0], width, label=labels[5], hatch = patterns[0], color = colors[0])
rects2 = ax.barh(numSubs, res[1], width, label=labels[4], hatch = patterns[1], color = colors[1])
rects3 = ax.barh(numSubs + width, res[2], width, label=labels[3], hatch = patterns[2], color = colors[2])

# for ii in range(1):
# 	ax.plot(numSubs, res[ii*3], 'o', linestyle='dotted', color = colors[ii], label = labels[ii*3])
# 	ax.plot(numSubs, res[ii*3+1], '*', linestyle='dashed', color = colors[ii+1], label = labels[ii*3+1])
# 	ax.plot(numSubs, res[ii*3+2], '.', linestyle='solid', color = colors[ii+2], label = labels[ii*3+2])

ax.bar_label(rects1, padding=5)
ax.bar_label(rects2, padding=5)
ax.bar_label(rects3, padding=5)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc = 'center', framealpha = 0.2, bbox_to_anchor=(1.2, 0.5))
ax.figure.set_size_inches(10, 5)
ax.set_ylabel(' Number of channel ')
ax.set_xlabel(' Accuracy (%) ')
# ax.set_xlim([70, 105])
# ax.set_ylim([0, 105])
plt.yticks(numSubs)
# plt.show()
plt.savefig("phi_channel.png")

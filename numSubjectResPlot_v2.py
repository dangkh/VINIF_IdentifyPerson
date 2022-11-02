import os 
import numpy as np 
import matplotlib.pyplot as plt
import random

numSubs = np.asarray([x * 10 + 10 for x in range(10)])

patterns = ["x", "|", "//" ]

res = [
[  99.94   , 99.91   , 99.92   , 99.89   , 99.44   , 99.34   , 99.25   , 99.30   , 99.30 , 99.30  ], 
[  99.98   , 99.90   , 99.91   , 99.89   , 99.50   , 99.35   , 99.26   , 99.31   , 99.29 , 99.30  ],
[  96.37   , 96.79   , 94.37   , 92.95   , 91.76   , 90.74   , 89.26   , 87.74   , 87.89 , 86.5  ]]

# print(res)
fig, ax = plt.subplots()
fig.suptitle('Accuracy with different number of subject', fontsize=11)
colors = ['limegreen', 'moccasin', 'thistle']
width = 3.1
labels = ['SVM + PSD + Z-score', 'SVM + PSD + EA', 'SVM + PSD + DEA', 'SVM + APF + Z-score', 'SVM + APF + EA', 'SVM + APF + DEA',
'CNN_LSTM + RAW + Z-score', 'CNN_LSTM + RAW + EA', 'CNN_LSTM + RAW + DEA']
rects1 = ax.barh(numSubs - width, res[0], width, label=labels[3], hatch = patterns[0], color = colors[0])
rects2 = ax.barh(numSubs, res[1], width, label=labels[4], hatch = patterns[1], color = colors[1])
rects3 = ax.barh(numSubs + width, res[2], width, label=labels[5], hatch = patterns[2], color = colors[2])

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
ax.figure.set_size_inches(12, 7)
ax.set_ylabel(' Number of subject ')
ax.set_xlabel(' Accuracy (%) ')
ax.set_xlim([70, 110])
ax.set_ylim([0, 120])
plt.yticks(numSubs)
# plt.show()
plt.savefig("phi_subject.png")

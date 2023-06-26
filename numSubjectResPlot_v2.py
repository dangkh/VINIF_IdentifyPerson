import os 
import numpy as np 
import matplotlib.pyplot as plt
import random

numSubs = np.asarray([x * 10 + 10 for x in range(10)])

patterns = ["*", ".", "1", "8", "+", "v" ]


# res = [
# [  95.11	,93.90	,90.61	,83.06	,83.82	,81.98	,82.41	,82.11	,82.86], 
# [  96.63	,96.20	,94.73	,93.27	,91.95	,90.70	,90.19	,90.02	,90.07],
# # [  95.16	,94.95	,91.92	,90.25	,89.41	,89.23	,87.72	,85.84	,85.47],

# # [  99.60	,99.24	,99.10	,98.17	,97.53	,97.17	,97.10	,96.72	,96.40],
# [  97.41	,97.08	,95.60	,97.00	,98.86	,96.62	,97.57	,97.37	,90.93],
# [  97.99	,97.61	,96.58	,96.39	,98.47	,96.98	,97.30	,97.17	,97.97],
# [  98.21	,97.24	,95.09	,96.10	,98.59	,97.12	,97.20	,97.39	,94.92],
# [  99.02	,99.80	,98.38	,96.47	,98.64	,97.24	,97.39	,97.68	,97.58]
# ]



res = [
[  93.83	,91.66	,88.89	,86.51	,84.65	,84.07	,83.02	,81.11	,80.81 ], 
[  99.07	,99.17	,98.93	,98.48	,97.85	,97.46	,97.19	,96.85	,96.27],
# [  95.16	,94.95	,91.92	,90.25	,89.41	,89.23	,87.72	,85.84	,85.47],

# [  99.60	,99.24	,99.10	,98.17	,97.53	,97.17	,97.10	,96.72	,96.40],
[  98.50	,99.11	,99.04	,98.78	,98.83	,98.44	,98.33	,98.73	,98.86],
[  99.57	,99.11	,99.80	,99.55	,99.47	,99.24	,99.27	,99.09	,99.13],
[  96.37	,96.79	,94.37	,92.95	,91.76	,90.74	,89.26	,87.74	,87.89],
[  99.94	,99.91	,99.92	,99.89	,99.44	,99.34	,99.25	,99.30	,99.30]
]

# print(res)
fig, ax = plt.subplots()
fig.suptitle('Accuracy with different number of subject for PHY dataset', fontsize=11)
colors = ['limegreen', 'moccasin', 'thistle']
width = 3.1
labels = ['SVM + PSD + Z-score', 'SVM + PSD + DDR',
'CNN + RAW + Z-score', 'CNN + RAW + DDR',
'SVM + AEF + Z-score', 'SVM + AEF + DDR']
# rects1 = ax.barh(numSubs - width, res[0], width, label=labels[5], hatch = patterns[0], color = colors[0])
# rects2 = ax.barh(numSubs, res[1], width, label=labels[4], hatch = patterns[1], color = colors[1])
# rects3 = ax.barh(numSubs + width, res[2], width, label=labels[3], hatch = patterns[2], color = colors[2])
for i in range(6):
	plt.plot([(x+1)*10 for x in range(0, 9)],res[i],marker=patterns[i], label = labels[i])

# plt.plot([x for x in range(0, 9)],res[1],label = labels[1])
# # plt.plot([x for x in range(0, 9)],res[2],label = labels[2])
# # plt.plot([x for x in range(0, 9)],res[3],label = labels[3])
# plt.plot([x for x in range(0, 9)],res[4],label = labels[4])
# plt.plot([x for x in range(0, 9)],res[5],label = labels[5])
# plt.plot([x for x in range(0, 9)],res[6],label = labels[6])
# plt.plot([x for x in range(0, 9)],res[7],label = labels[7])

# for ii in range(1):
# 	ax.plot(numSubs, res[ii*3], 'o', linestyle='dotted', color = colors[ii], label = labels[ii*3])
# 	ax.plot(numSubs, res[ii*3+1], '*', linestyle='dashed', color = colors[ii+1], label = labels[ii*3+1])
# 	ax.plot(numSubs, res[ii*3+2], '.', linestyle='solid', color = colors[ii+2], label = labels[ii*3+2])

# ax.bar_label(rects1, padding=5)
# ax.bar_label(rects2, padding=5)
# ax.bar_label(rects3, padding=5)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(loc = 'center', framealpha = 0.2, bbox_to_anchor=(1.2, 0.5))
ax.figure.set_size_inches(10, 5)
ax.set_ylabel(' Accuracy (%) ')
ax.set_xlabel(' Number of subject')
# ax.set_xlim([70, 105])
ax.set_ylim([80, 101])
# plt.yticks(numSubs)
# plt.show()
plt.savefig("phy_subject.eps", format='eps')

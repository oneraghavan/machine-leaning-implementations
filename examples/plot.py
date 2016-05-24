# import matplotlib.pyplot as plt
# import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
#
# x = np.array(range(-150,150,2))
# log_y = np.log(x)
# linear_y = x + 2
# sigmoid_y = []
#
# for i in x:
#   sigmoid_y.append(sigmoid(i))
#
# f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
# ax1.plot(x, log_y)
# ax2.plot(x, linear_y)
# ax3.plot(x, sigmoid_y)
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0)
# # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#
# plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
# plt.show()

import numpy as np
import pylab as plt

X = np.arange(-100., 100., 0.005)
X_2 = np.arange(-500., 500., 0.005)
Y1 = np.log(X)
Y2 = 9 * X + 6
Y3 = []

for i in X_2:
  Y3.append(sigmoid(i))


plt.plot(X,Y1,color='b')
plt.plot(X,Y2,color='g')
plt.plot(X_2,Y3,color='r')
plt.show()
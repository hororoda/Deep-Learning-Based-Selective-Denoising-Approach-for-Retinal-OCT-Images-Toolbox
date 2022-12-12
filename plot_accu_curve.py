"""
plot accuracy curves
.csv files can be downloaded from tensorboard
"""


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator


# load data
# accuracy curve for training
train_accu = pd.read_csv("")
# accuracy curve for validation
val_accu = pd.read_csv("")


# plot
step1 = train_accu["Step"].values.tolist()
accu1 = train_accu["Value"].values.tolist()
step2 = val_accu["Step"].values.tolist()
accu2 = val_accu["Value"].values.tolist()

plt.figure(figsize=(8, 6), dpi=100)

# labels
plt.plot(step1, accu1, label='train')
plt.legend(loc=4, labelspacing=0.5, handlelength=3, fontsize=20, shadow=False)
plt.plot(step2, accu2, label='val')
plt.legend(loc=4, labelspacing=0.5, handlelength=3, fontsize=20, shadow=False)

# axises
x_major_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(0.1)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax.set_xlabel(xlabel='epoch', fontsize=20)
ax.set_ylabel(ylabel='accuracy', fontsize=20)

plt.xlim(0, 24)
plt.ylim(0.5, 1.05)

plt.show()

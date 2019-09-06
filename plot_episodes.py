import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


csv_file = '2019-09-06_10-37-46_Reacher-v1_eval.csv'

X = genfromtxt('results/'+csv_file)
print(X)


plt.plot(X[:, 0], X[:, 2])
plt.xlabel('Episode number')
plt.ylabel('Reward')
# plt.show()
plt.savefig('plots/'+csv_file[:-4]+'.png')




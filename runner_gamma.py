from scipy.io import loadmat
import pandas as pd
import numpy as np
from CJHawkes import clusterhawkes
import matplotlib.cm as cm
import matplotlib.pyplot as plt



W = np.array(pd.read_csv('synthetic_gamma.csv'))
X = loadmat('X_hawkes_syn_gamma.mat').get('X') # mat file dimension: #. student x #. time stamps x #. assignments; e.g. X[i,:,j]: sequence of student-assignment pair (s_i, a_j)
itermax = 30
W_learn, funcVal, M_learn, U_learn = clusterhawkes(X, 15, 0.1, 3, 0.7, itermax, bFlag= 2,prior = 'gamma',gamma_theta=[1/5,1/5,1/5],gamma_k= [2,3,4]) # learned parms
clus_num = 3

clus_task_num = 30
comm_dim = 1
dimension = 60
task_num = clus_num * clus_task_num  # total task number.
# independent dimension for all tasks.
clus_dim = np.floor((dimension - comm_dim) / 2)  # dimension of cluster
sample_size = 150
OrderedLearnModel = np.zeros(W_learn.shape)
for i in range(1, clus_num + 1):
    clusModel = W_learn[:, [x - 1 for x in range(i, task_num + 1, clus_num)]]
    OrderedLearnModel[:, range((i - 1) * clus_task_num, i * clus_task_num)] = clusModel
corr = 1 - np.corrcoef(OrderedLearnModel.T)
plt.imshow(corr, cmap=cm.gray)
plt.title('learned model without gamma prior')
plt.show()


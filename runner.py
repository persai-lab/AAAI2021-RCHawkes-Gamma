from scipy.io import loadmat
import pandas as pd
import numpy as np
from cluster_hawkes_general import clusterhawkes
import matplotlib.cm as cm
import matplotlib.pyplot as plt


W = np.array(pd.read_csv('synthetic_alpha.csv'))
X = loadmat('X_hawkes_syn.mat').get('X')
#
W_learn, funcVal, M_learn, U_learn = clusterhawkes(X, 10, 0.1, 4,0.7, 30,'gamma',2,2)
#
# # synethic data info
clus_mean = 0.4
clus_var = 0.85  # cluster variance
task_var = 0.05  # inter task variance
nois_var = 0.01  # variance of noise
clus_num = 4  # clusters
clus_task_num = 50
task_num = clus_num * clus_task_num
sample_size = 100
dimension = 20  # total dimension
comm_dim = 2
hawkes_beta = 0.7
# hawkes_mu = 0.1
#
OrderedLearnModel = np.zeros(W_learn.shape)
for i in range(1, clus_num + 1):
    clusModel = W_learn[:, [x - 1 for x in range(i, task_num + 1, clus_num)]]
    OrderedLearnModel[:, range((i - 1) * clus_task_num, i * clus_task_num)] = clusModel
corr = 1 - np.corrcoef(OrderedLearnModel.T)
print(W_learn)
print(W_learn.shape)
print(U_learn)
plt.imshow(corr, cmap=cm.gray)
plt.title('learned model')
plt.show()
# pd.DataFrame(W_learn).to_csv('W_learn.csv',index = False)

# OrderedTrueModel = np.zeros(W.shape)
# for i in range(1, clus_num + 1):
#     clusModel = W[:, [x - 1 for x in range(i, task_num + 1, clus_num)]]
#     OrderedTrueModel[:, range((i - 1) * clus_task_num, i * clus_task_num)] = clusModel
# # corr = 1 - np.corrcoef(OrderedLearnModel.T)
#
# plt.imshow(1 - np.corrcoef(OrderedTrueModel.T), cmap=cm.gray)
# plt.title('true model')
# plt.show()
# print(W_learn,U_learn)
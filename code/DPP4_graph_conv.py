# -*- coding: utf-8 -*-
"""
Reproduce the Merck DPP4 experiment described in section 4.1 of the paper:

"A generalization of Convolutional Neural Networks to Graph-Structured Data"

The Merck challenge data was downloaded from the supplementary
material to the paper "Deep Neural Nets as a Method for Quantitative 
Structure–Activity Relationships" by Ma et al., located at:
http://pubs.acs.org/doi/suppl/10.1021/ci500747n
"""

### Dependencies 
import os
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import adam, RMSprop
from keras.regularizers import l2, l1

from graph_convolution import GraphConv
from dcnn import DiffusionConv
from dcnn_modified import DiffusionConvModified

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

import cPickle as pickle

seed_val = 1984
np.random.seed(seed_val) # for reproducibility
#tf.set_random_seed(seed_val)
print('Random seed = %d'%seed_val)

###
def r_square_np(y_true, y_pred):
    '''
    Calcualte the R^2 coefficent between y_true and y_pred
    '''
    y_pred_mean = np.mean(y_pred)
    y_true_mean = np.mean(y_true)
    square_corr_values = np.square((np.sum((y_pred-y_pred_mean)*(y_true-y_true_mean)))/(np.sqrt(np.sum(np.square(y_pred-y_pred_mean))*np.sum(np.square(y_true-y_true_mean)))))
    return square_corr_values

### Parameters
batch_size=200
epochs= 40
num_neighbors= 5
filters_1 = 10
filters_2 = 20
num_hidden_1 = 300
num_hidden_2 = 100
#PerformDiffusion = bool(int(sys.argv[1]))
#ConvWithSimValuesAndWeights = bool(int(sys.argv[1]))
K = int(sys.argv[1])
#print('PerformDiffusion=', PerformDiffusion)
#print('ConvWithSimValuesAndWeights=', ConvWithSimValuesAndWeights)
results = dict()

# %% 
### Load the data
tr_loc = '../data/DPP4_training_disguised.csv'
val_loc = '../data/DPP4_test_disguised.csv'

tr_data = pd.read_csv(open(tr_loc,'r'))
val_data = pd.read_csv(open(val_loc,'r'))

features_names = np.intersect1d(tr_data.columns.values[2:],
                                val_data.columns.values[2:])

X_train, y_train = [np.array(tr_data[features_names],dtype = 'float32'),
                    np.array(tr_data['Act'],dtype = 'float32')]

active_ix = np.array(X_train,dtype='bool').sum(0) >= 20
X_train = X_train[:,active_ix]
X_train = (X_train / X_train.max(0))

print('Training data shape:(%d,%d)'%(X_train.shape))
                    
X_test, y_test = [np.array(val_data[features_names],dtype = 'float32'),
                    np.array(val_data['Act'],dtype = 'float32')]

X_test = X_test[:,active_ix]
X_test = (X_test / X_test.max(0))

print('Test data shape:(%d,%d)'%(X_test.shape))

### Prepare the Graph Correlation matrix 
corr_mat = np.array(normalize(np.abs(np.corrcoef(X_train.transpose())), norm='l1', axis=1),dtype='float64')
#print(corr_mat)
#print(np.allclose(corr_mat, corr_mat.T, atol=1e-8)) # Checking if corr_mat is symmetric.
#print(np.sum(corr_mat, axis=1))


# Calculate Q matrix using random walk
#P = np.add(corr_mat, np.eye(corr_mat.shape[0]))
#Q = P
#prob_transition_tensor.append(np.eye(corr_mat.shape[0]))
#prob_transition_tensor.append(corr_mat)
#P = corr_mat
Q = np.zeros(corr_mat.shape) # Q matrix defined in Hechtlinger's paper.
prob_transition_tensor = list() # Initialize Probability Transition Tensor, used in Atwood's paper.
for i in range(K):
    #P = np.matmul(P,P)
    P = np.linalg.matrix_power(corr_mat, i)
    prob_transition_tensor.append(P)
    #if PerformDiffusion == True:
    Q = Q + P
#print(Q)
    
corr_mat = Q
prob_transition_tensor = np.array(prob_transition_tensor, dtype='float32')
prob_transition_tensor = np.transpose(prob_transition_tensor, [1,0,2])
print('Shape of Prob_transition_tensor:',prob_transition_tensor.shape)
graph_mat = np.argsort(corr_mat,1)[:,-num_neighbors:]
cor_graph_mat = np.sort(corr_mat,1)[:,-num_neighbors:]

print('graph_mat shape:(%d,%d)'%(graph_mat.shape))

# %%

### 1 layer GCNN
g_model = Sequential()
g_model.add(GraphConv(filters=filters_1, neighbors_ix_mat = graph_mat, 
                      num_neighbors=num_neighbors, activation='relu', 
                      input_shape=(X_train.shape[1],1),
                      correlation_graph_mat = cor_graph_mat,
                      ConvWithSimValuesAndWeights=False))
g_model.add(Dropout(0.25))
g_model.add(Flatten())
g_model.add(Dense(1, kernel_regularizer=l2(0.01)))
g_model.add(Dropout(0.1))

g_model.summary()

g_model.compile(loss='mean_squared_error', optimizer='adam')

results['g'] = []
for i in range(epochs):
    g_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
              epochs=1,
              batch_size=batch_size,
              verbose = 0,)

    y_pred = g_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
                             batch_size=100).flatten()
    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
    results['g'].append(r_squared)
    print('Epoch: %d, R squared: %.5f'%(i,r_squared))

results['g'] = np.array(results['g'])
print('1-Conv R squared = %.5f'%results['g'][-1])

# %%

### 1 layer GCNN with Similarity Values and Weights
g_with_sim_model = Sequential()
g_with_sim_model.add(GraphConv(filters=filters_1, neighbors_ix_mat = graph_mat, 
                      num_neighbors=num_neighbors, activation='relu', 
                      input_shape=(X_train.shape[1],1),
                      correlation_graph_mat = cor_graph_mat,
                      ConvWithSimValuesAndWeights=True))
g_with_sim_model.add(Dropout(0.25))
g_with_sim_model.add(Flatten())
g_with_sim_model.add(Dense(1, kernel_regularizer=l2(0.01)))
g_with_sim_model.add(Dropout(0.1))

g_with_sim_model.summary()

g_with_sim_model.compile(loss='mean_squared_error', optimizer='adam')

results['g_wsim'] = []
for i in range(epochs):
    g_with_sim_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
              epochs=1,
              batch_size=batch_size,
              verbose = 0,)

    y_pred = g_with_sim_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
                             batch_size=100).flatten()
    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
    results['g_wsim'].append(r_squared)
    print('Epoch: %d, R squared: %.5f'%(i,r_squared))

results['g_wsim'] = np.array(results['g_wsim'])
print('1-Conv R squared = %.5f'%results['g_wsim'][-1])

# %%

### 1 layer DCNN
dc_model = Sequential()
dc_model.add(DiffusionConv(filters=filters_1, prob_transition_tensor = prob_transition_tensor,
                        num_hops=K, activation='relu',
                        input_shape=(X_train.shape[1],1)))
dc_model.add(Dropout(0.25))
dc_model.add(Flatten())
dc_model.add(Dense(1, kernel_regularizer=l2(0.01)))
dc_model.add(Dropout(0.1))

dc_model.summary()

dc_model.compile(loss='mean_squared_error', optimizer='adam')

results['dc'] = []
for i in range(epochs):
    dc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
              epochs=1,
              batch_size=batch_size,
              verbose = 0,)

    y_pred = dc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
                             batch_size=100).flatten()
    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
    results['dc'].append(r_squared)
    print('Epoch: %d, R squared: %.5f'%(i,r_squared))

results['dc'] = np.array(results['dc'])
print('1-Conv R squared = %.5f'%results['dc'][-1])

# %%

### 1 layer DCNN_Modified
dc_mfd_model = Sequential()
dc_mfd_model.add(DiffusionConvModified(filters=filters_1, prob_transition_tensor = prob_transition_tensor,
                        num_hops=K, activation='relu',
                        input_shape=(X_train.shape[1],1)))
dc_mfd_model.add(Dropout(0.25))
dc_mfd_model.add(Flatten())
dc_mfd_model.add(Dense(1, kernel_regularizer=l2(0.01)))
dc_mfd_model.add(Dropout(0.1))

dc_mfd_model.summary()

dc_mfd_model.compile(loss='mean_squared_error', optimizer='adam')

results['dc_mfd'] = []
for i in range(epochs):
    dc_mfd_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
              epochs=1,
              batch_size=batch_size,
              verbose = 0,)

    y_pred = dc_mfd_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
                             batch_size=100).flatten()
    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
    results['dc_mfd'].append(r_squared)
    print('Epoch: %d, R squared: %.5f'%(i,r_squared))

results['dc_mfd'] = np.array(results['dc_mfd'])
print('1-Conv R squared = %.5f'%results['dc_mfd'][-1])

# %%

#### 1 layer GCNN + 1 layer FC
#g_fc_model = Sequential()
#g_fc_model.add(GraphConv(filters=filters_1, neighbors_ix_mat = graph_mat, 
#                      num_neighbors=num_neighbors, activation='relu', 
#                      input_shape=(X_train.shape[1],1),
#                      correlation_graph_mat = cor_graph_mat,
#                      ConvWithSimValuesAndWeights = ConvWithSimValuesAndWeights))
#g_fc_model.add(Dropout(0.25))
#g_fc_model.add(Flatten())
#g_fc_model.add(Dense(num_hidden_2, activation='relu', kernel_regularizer=l2(0.01),))
#g_fc_model.add(Dropout(0.25))
#g_fc_model.add(Dense(1, kernel_regularizer=l2(0.01)))
#g_fc_model.add(Dropout(0.1))
#
#g_fc_model.summary()
#
#g_fc_model.compile(loss='mean_squared_error', optimizer='adam')
#
#results['g_fc'] = []
#for i in range(epochs):
#    g_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), 
#                   y_train,
#                   epochs=1,
#                   batch_size=batch_size,
#                   verbose = 0,)
#
#    y_pred = g_fc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
#                                batch_size=100).flatten()
#    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
#    results['g_fc'].append(r_squared)
#    print('Epoch: %d, R squared: %.5f'%(i,r_squared))
#
#results['g_fc'] = np.array(results['g_fc'])
#print('Conv-FC R squared = %.5f'%results['g_fc'][-1])
#
## %%
#
#### 1 layer GCNN with Similarity Values and Weights + 1 layer FC
#g_with_sim_fc_model = Sequential()
#g_with_sim_fc_model.add(GraphConv(filters=filters_1, neighbors_ix_mat = graph_mat, 
#                      num_neighbors=num_neighbors, activation='relu', 
#                      input_shape=(X_train.shape[1],1),
#                      correlation_graph_mat = cor_graph_mat,
#                      ConvWithSimValuesAndWeights=True))
#g_with_sim_fc_model.add(Dropout(0.25))
#g_with_sim_fc_model.add(Flatten())
#g_with_sim_fc_model.add(Dense(num_hidden_2, activation='relu', kernel_regularizer=l2(0.01),))
#g_with_sim_fc_model.add(Dropout(0.25))
#g_with_sim_fc_model.add(Dense(1, kernel_regularizer=l2(0.01)))
#g_with_sim_fc_model.add(Dropout(0.1))
#
#g_with_sim_fc_model.summary()
#
#g_with_sim_fc_model.compile(loss='mean_squared_error', optimizer='adam')
#
#results['g_wsim_fc'] = []
#for i in range(epochs):
#    g_with_sim_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
#              epochs=1,
#              batch_size=batch_size,
#              verbose = 0,)
#
#    y_pred = g_with_sim_fc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
#                             batch_size=100).flatten()
#    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
#    results['g_wsim_fc'].append(r_squared)
#    print('Epoch: %d, R squared: %.5f'%(i,r_squared))
#
#results['g_wsim_fc'] = np.array(results['g_wsim_fc'])
#print('1-Conv R squared = %.5f'%results['g_wsim_fc'][-1])
#
## %%
#
#### 1 layer DCNN + 1 layer FC
#dc_fc_model = Sequential()
#dc_fc_model.add(DiffusionConv(filters=filters_1, prob_transition_tensor = prob_transition_tensor,
#                        num_hops=K, activation='relu',
#                        input_shape=(X_train.shape[1],1)))
#dc_fc_model.add(Dropout(0.25))
#dc_fc_model.add(Flatten())
#dc_fc_model.add(Dense(num_hidden_2, activation='relu', kernel_regularizer=l2(0.01),))
#dc_fc_model.add(Dropout(0.25))
#dc_fc_model.add(Dense(1, kernel_regularizer=l2(0.01)))
#dc_fc_model.add(Dropout(0.1))
#
#dc_fc_model.summary()
#
#dc_fc_model.compile(loss='mean_squared_error', optimizer='adam')
#
#results['dc_fc'] = []
#for i in range(epochs):
#    dc_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
#              epochs=1,
#              batch_size=batch_size,
#              verbose = 0,)
#
#    y_pred = dc_fc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
#                             batch_size=100).flatten()
#    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
#    results['dc_fc'].append(r_squared)
#    print('Epoch: %d, R squared: %.5f'%(i,r_squared))
#
#results['dc_fc'] = np.array(results['dc_fc'])
#print('1-Conv R squared = %.5f'%results['dc_fc'][-1])
#
## %%
#
#### 1 layer DCNN_Modified + 1 layer FC
#dc_mfd_fc_model = Sequential()
#dc_mfd_fc_model.add(DiffusionConvModified(filters=filters_1, prob_transition_tensor = prob_transition_tensor,
#                        num_hops=K, activation='relu',
#                        input_shape=(X_train.shape[1],1)))
#dc_mfd_fc_model.add(Dropout(0.25))
#dc_mfd_fc_model.add(Flatten())
#dc_mfd_fc_model.add(Dense(num_hidden_2, activation='relu', kernel_regularizer=l2(0.01),))
#dc_mfd_fc_model.add(Dropout(0.25))
#dc_mfd_fc_model.add(Dense(1, kernel_regularizer=l2(0.01)))
#dc_mfd_fc_model.add(Dropout(0.1))
#
#dc_mfd_fc_model.summary()
#
#dc_mfd_fc_model.compile(loss='mean_squared_error', optimizer='adam')
#
#results['dc_mfd_fc'] = []
#for i in range(epochs):
#    dc_mfd_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), y_train,
#              epochs=1,
#              batch_size=batch_size,
#              verbose = 0,)
#
#    y_pred = dc_mfd_fc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1), 
#                             batch_size=100).flatten()
#    r_squared = (np.corrcoef(y_pred,y_test)**2)[0,1]
#    results['dc_mfd_fc'].append(r_squared)
#    print('Epoch: %d, R squared: %.5f'%(i,r_squared))
#
#results['dc_mfd_fc'] = np.array(results['dc_mfd_fc'])
#print('1-Conv R squared = %.5f'%results['dc_mfd_fc'][-1])
#
# %%

# Generate Figure
legend_dict = {'g': '$GC_{'+str(filters_1)+'}$',
               'g_wsim': '$GCWSIM_{'+str(filters_1)+'}$',
               'dc': '$DC_{'+str(filters_1)+'}$',
               'dc_mfd': '$DCMFD_{'+str(filters_1)+'}$',
                }

x = np.arange(0, epochs)

sns.set_style("whitegrid",{'legend.frameon': True})
sns.set_context("paper")
sns.set_palette("hls")

plt.figure(figsize=(12, 9))

plt.plot(x, results['g'], label = legend_dict['g'], marker = 's', markersize = 8, linewidth=2)
plt.plot(x, results['g_wsim'], label = legend_dict['g_wsim'], marker = '8', markersize = 8, linewidth = 2)
plt.plot(x, results['dc'], label = legend_dict['dc'], marker = '^', markersize = 8, linewidth = 2)
plt.plot(x, results['dc_mfd'], label = legend_dict['dc_mfd'], marker = 'D', markersize = 8, linewidth = 2)

plt.title('Convergence Curves', fontsize = 'xx-large')
plt.xlabel('Epoch', fontsize = 'x-large')
plt.ylabel(r'$R^{2}$', fontsize = 'xx-large')
plt.tick_params(axis='x', labelsize='x-large')
plt.tick_params(axis='y', labelsize='x-large')

fontP = FontProperties()
fontP.set_size('xx-large')
legend = plt.legend(title='Architecture', fontsize='x-large', loc = 'lower center', 
           fancybox=True, shadow=True, ncol = 2, prop = fontP)
plt.setp(legend.get_title(),fontsize='x-large')
sns.axes_style()

if not os.path.exists('results/'):
    os.makedirs('results/')

plt.savefig('results/plot_graph_conv_K='+str(K)+'.png', bbox_inches='tight')

#If needed dump the results
#pickle.dump(results, open('results/DPP4_conv_results_final.p','wb'))


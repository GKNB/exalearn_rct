#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import inspect
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import io, os, sys
from torch import FloatTensor
import itertools
from sklearn.metrics import confusion_matrix, mean_squared_error
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
import time


# In[2]:


SEQ_LENGTH = 256
STRIDE = 1
NUM_BATCHES = 120
HIDDEN_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT = 0.2
NUM_OUTPUTS = 2
NUM_LAYERS = 3
BATCH_SIZE=512


# In[3]:


class Lenet1d( torch.nn.Module ):
    def __init__(self, len_input, 
                 conv1=(16, 3, 1), 
                 pool1=(2, 2), 
                 conv2=(32, 4, 2), 
                 pool2=(2, 2), 
                 fc1=256, 
                 num_classes=3):
        super(Lenet1d, self).__init__()
        
        n = len_input
        # In-channels, Out-channels, Kernel_size, stride ...
        self.conv1 = torch.nn.Conv1d(1, conv1[0], conv1[1], stride=conv1[2])
        n = (n - conv1[1]) // conv1[2] + 1
        
        self.pool1 = torch.nn.MaxPool1d(pool1[0], stride=pool1[1] )
        n = (n - pool1[0]) // pool1[1] + 1
        
        self.conv2 = torch.nn.Conv1d(conv1[0], conv2[0], conv2[1], stride=conv2[2])
        n = (n - conv2[1]) // conv2[2] + 1
        
        self.pool2 = torch.nn.MaxPool1d(pool2[0], stride=pool2[1] )
        n = (n - pool2[0]) // pool2[1] + 1
        
        self.features = torch.nn.Sequential( self.conv1, self.pool1, self.conv2, self.pool2 )
        
        self.fc1 = torch.nn.Linear(n*conv2[0], fc1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(fc1, num_classes)        
        self.prediction = torch.nn.Sequential( self.fc1, self.relu, self.fc2 )            
    
    def fingerprint(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        return x
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)
        return x


# In[4]:


class XFerLearning(torch.nn.Module):
    def __init__(self, num_hidden, num_output):
        super(XFerLearning, self).__init__()
        self.regression_layer=torch.nn.Linear(num_hidden, num_output)
    
    def forward(self, X):
        return self.regression_layer(X)    


# In[5]:


# motivated out of https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize
# Window len = L, Stride len/stepsize = S
def strided_app(a, L, S ):  
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def batch_stride(X, L, S):
    nrows = ((X.shape[1]-L)//S)+1
    X_strided = np.empty(shape=(X.shape[0],L, nrows))
    for i in range(0, X.shape[0]-1):
        X_strided[i,:,:]=(strided_app(X[i,:],L,S)).T
    return X_strided
#https://stackoverflow.com/questions/43870647/how-to-duplicate-a-row-or-column-in-a-numpy-array
def dup_cols(a, indx, num_dups=1):
    return np.insert(a,[indx+1]*num_dups,a[:,[indx]],axis=1)

#https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


# # load the all_in_one data

# In[6]:


#X_scaled = np.load('./multitask_all_Y_balanced.npy')
#y_scaled = np.load('./multitask_all_P_balanced.npy')

X_scaled = np.load('/home/twang3/myWork/exalearn_project/run_theta/baseline/ml/Output-all-Y.npy')
y_scaled = np.load('/home/twang3/myWork/exalearn_project/run_theta/baseline/ml/Output-all-P.npy')

print(X_scaled.shape, y_scaled.shape)


# In[7]:


X_scaled = np.float32(X_scaled)
y_scaled = np.float32(y_scaled)
train_idx, test_idx = train_test_split(range(len(X_scaled)), test_size=0.05, random_state=42)
test_idx.sort()
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# # choose the training device

# In[8]:


num_gpus = 0
multigpu = False
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
if device is 'cuda':
    num_gpus = torch.cuda.device_count()
    if  num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        multigpu = False


# In[9]:


X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train).reshape(-1,y_train.shape[1])
X_train_torch = FloatTensor(X_train_torch)
y_train_torch = FloatTensor(y_train_torch)

X_test_torch = torch.from_numpy(X_test)#.cuda()
y_test_torch = torch.from_numpy(y_test).reshape(-1,y_train.shape[1])#.cuda()
X_test_torch = FloatTensor(X_test_torch)
y_test_torch = FloatTensor(y_test_torch)
X_test_torch = X_test_torch.to(device)
y_test_torch = y_test_torch.to(device)

X_train_torch = X_train_torch.reshape((X_train_torch.shape[0], 1, X_train_torch.shape[1]))
X_test_torch = X_test_torch.reshape((X_test_torch.shape[0], 1, X_test_torch.shape[1]))

print(X_train_torch.shape,y_train_torch.shape,X_test_torch.shape, y_test_torch.shape)
#sampler = WeightedRandomSampler(unique_sample_weights, BATCH_SIZE)

dataset = TensorDataset(X_train_torch,y_train_torch)
batch_loader = DataLoader(dataset, batch_size=BATCH_SIZE) #sampler=sampler)


# In[10]:


lenet_trained_model = Lenet1d(2806, num_classes=3)
#lenet_trained_model.load_state_dict(torch.load('../../pytorch_classifier/IEEEBigData_fingerprint_model.pt', map_location=torch.device('cpu')))
if multigpu:
    #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    trained_model = nn.DataParallel(rnn)
lenet_trained_model = lenet_trained_model.to(device)
simple_xfer_model = XFerLearning(256,3)
simple_xfer_model = simple_xfer_model.to(device)
optimizer1 = torch.optim.Adam(simple_xfer_model.parameters(), lr=LEARNING_RATE)
optimizer2 = torch.optim.Adam(lenet_trained_model.parameters(), lr=LEARNING_RATE)
criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.BCEWithLogitsLoss()


# In[11]:


print("the number of cpu threads: {}".format(torch.get_num_threads()))
t = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
#     rand_idxs = np.random.permutation(X_train.shape[0])
#     batches = np.array_split(rand_idxs, NUM_BATCHES)
    print("Epoch: {:4}".format(epoch))
    #simple_xfer_model.train()
    lenet_trained_model.train()
    for batch_idx, current_batch in enumerate(batch_loader):        
        optimizer1.zero_grad()        
        optimizer2.zero_grad()
#         temp_cubic_idxs = np.random.permutation(X_cubic.shape[0])
#         temp_cubic_idxs = temp_cubic_idxs[0:100]
#         current_X_cubic = X_cubic_torch[temp_cubic_idxs,:,:]
#         current_y_cubic = y_cubic_torch[temp_cubic_idxs,:]
#         inp = torch.cat((current_batch[0],current_X_cubic))
        inp = current_batch[0].to(device)
        #with torch.no_grad():
        xfer_features = lenet_trained_model.fingerprint(inp)
        #xfer_features = lenet_trained_model.fingerprint(inp)
        class_output = lenet_trained_model.fc2(xfer_features)
        regression_output = simple_xfer_model(xfer_features)        
        #current_batch_y = (y_train_torch[batch]).cuda()
        current_batch_y = current_batch[1].to(device)
        #current_batch_y = current_batch[1].cuda()
#         current_batch_y=torch.cat((current_batch[1],current_y_cubic))
#         current_batch_y = current_batch_y.cuda()
        regression_gndtruth = current_batch_y[:,0:3]
        loss1 = criterion1(regression_output, regression_gndtruth)
        loss1.backward(retain_graph=True)
        class_gndtruth = current_batch_y[:,3:6]
        loss2 = criterion2(class_output, class_gndtruth )
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        elapsed = time.time() - t
        if (batch_idx % 20 == 0):
            print(" Time Taken: {:15.6f}".format(elapsed/100))
            print("  Batch {:3}/{}: {:15.6f}".format(batch_idx, len(batch_loader), loss1.item()))
            print("  Batch {:3}/{}: {:15.6f}".format(batch_idx, len(batch_loader), loss2.item()))
            t = time.time()
#    if (epoch % 50 == 0):
#        simple_xfer_model.eval()
#        lenet_trained_model.eval()
#        with torch.no_grad():
#            xfer_features = lenet_trained_model.fingerprint(X_test_torch)            
#            regression_output = simple_xfer_model(xfer_features)
#            class_output = lenet_trained_model.fc2(xfer_features)
#        loss1 = criterion1(regression_output, y_test_torch[:,0:3].to(device))
#        #loss1 = criterion1(regression_output, y_test_torch[:,0:3].cuda())
#        print('50 epoch time: {:15.6f}'.format(elapsed/50))
#        print("  Loss: {:15.6f}".format(loss1.item()))
##         fig,ax = plt.subplots(figsize=(10,5), nrows=1,ncols=3)
##         ax[0].plot(y_test_torch.cpu().detach().numpy()[:,0],regression_output.cpu().detach().numpy()[:,0])
##         ax[0].plot(y_test_torch.cpu().detach().numpy()[:,0],y_test_torch.cpu().detach().numpy()[:,0])
##         ax[1].plot(y_test_torch.cpu().detach().numpy()[:,1],regression_output.cpu().detach().numpy()[:,1])
##         ax[1].plot(y_test_torch.cpu().detach().numpy()[:,1],y_test_torch.cpu().detach().numpy()[:,1])
##         ax[2].plot(y_test_torch.cpu().detach().numpy()[:,2],regression_output.cpu().detach().numpy()[:,2])
##         ax[2].plot(y_test_torch.cpu().detach().numpy()[:,2],y_test_torch.cpu().detach().numpy()[:,2])
##         plt.show()
#        y_test_torch_cpu = y_test_torch.cpu().detach().numpy()
#        pred_labels_cpu = class_output.cpu().detach().numpy()
##         print(y_test_torch_cpu.shape, pred_labels_cpu.shape)
##         print(y_test_torch_cpu[1:10,3:6], pred_labels_cpu[1:10,:])
#        gnd_labels = np.argmax(y_test_torch_cpu[:,3:6], axis=1)
#        pred_labels = np.argmax(pred_labels_cpu, axis=1)
##         print(gnd_labels.shape,pred_labels.shape)
##         print(gnd_labels[1:10],pred_labels[1:10])
#        cm = confusion_matrix(gnd_labels,pred_labels)
#        print(cm)
#    if (epoch % 50 == 0):
#        filename = 'models/mt_all_balanced_lenet_learning_regressor_'+ str(NUM_EPOCHS) + '_' + str(epoch) + '_epoch.pkl'
#        torch.save(simple_xfer_model.state_dict(), filename)
#        filename = 'models/mt_all_balanced_lenet_learning_lenet_'+ str(NUM_EPOCHS) + '_' + str(epoch) + '_epoch.pkl'
#        torch.save(lenet_trained_model.state_dict(), filename)


# In[ ]:





from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import h5py
import time


root_path = '/lus-projects/CSC249ADCD08/twang/real_work_theta/baseline-rct/sim/'
path_in_list = ['test_cubic/', 
                'test_trigonal_part1/', 
                'test_trigonal_part2/', 
                'test_tetragonal/']
filename_in_list = ['cubic_1001460_cubic_part', 
                    'trigonal_1522004_trigonal_part', 
                    'trigonal_1522004_trigonal_part',
                    'tetragonal_1531431_tetragonal_part']
rank_max_list = [64, 64, 64, 64]
#rank_max_list = [64, 512, 512, 512]

for si in range(4):

    rank_max = rank_max_list[si]
    path_in = path_in_list[si]
    filename_in = filename_in_list[si]

    for ri in range(0, rank_max):
        
        with h5py.File(root_path + path_in + filename_in + str(ri) + '.hdf5', 'r') as f:
    
            start = time.time()

            dhisto = f['histograms']
            histo_sub = dhisto[:]
            dparams = f['parameters']            
            params_sub = dparams[:]

            end = time.time()
            print("si = ", si, " ri = ", ri, " Running time p1 = ", end - start)
            
            start = time.time()
            if ri == 0:
                histo = histo_sub
                params = params_sub
    
            else:
                histo = np.concatenate((histo, dhisto[:]), axis=0)
                params = np.concatenate((params, dparams[:]), axis=0)
#                histo = np.vstack([histo, histo_sub])
#                params = np.vstack([params, params_sub])

            end = time.time()
            print("Running time p2 = ", end - start)


    print('Shape of histo read: ', histo.shape)
    print('Shape of parameters read: ', params.shape)
    
    scaler = MinMaxScaler(copy=True)
    # Establish X,y data
    
    X = histo[:, 1, :]
    y = params[:,0]
    print('Shape of X: ', X.shape)
    print('Shape of y: ', y.shape)
    # normalize data
    
    X_post = scaler.fit_transform(X.T)
    X_post = X_post.T
    print('Shape of X_post: ', X_post.shape)



#for i in range(0,64):
#    
#    with h5py.File(root_path + path_in_list[0] + filename_in_list[0] + str(i) + '.hdf5', 'r') as f:
#
#        dhisto = f['histograms']
#        dparams = f['parameters']
#        
#        if i == 0:
#            histo = dhisto[:]
#            params = dparams[:]
#
#        else:
#            histo = np.concatenate((histo, dhisto[:]), axis=0)
#            params = np.concatenate((params, dparams[:]), axis=0)


#print('Shape of histo read: ', histo.shape)
#
#print('Shape of parameters read: ', params.shape)
#
#scaler = MinMaxScaler(copy=True)
#
## Establish X,y data
#
#X = histo[:, 1, :]
#
#y = params[:,0]
#
#print('Shape of X: ', X.shape)
#
#print('Shape of y: ', y.shape)
#
## normalize data
#
#X_post = scaler.fit_transform(X.T)
#
#X_post = X_post.T
#
#print('Shape of X_post: ', X_post.shape)




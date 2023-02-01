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
filename_out_list = ['cubic',
                     'trigonal_p1',
                     'trigonal_p2',
                     'tetragonal']
#rank_max_list = [16, 16, 16, 16]
rank_max_list = [64, 512, 512, 512]

rank_in_max = 8

for si in range(4):

    rank_max = rank_max_list[si]
    path_in = path_in_list[si]
    filename_in = filename_in_list[si]
    filename_out = filename_out_list[si]

    for ri_out in range(0, int(rank_max / rank_in_max)):

        for ri_in in range(ri_out * rank_in_max, (ri_out + 1) * rank_in_max):
            
            with h5py.File(root_path + path_in + filename_in + str(ri_in) + '.hdf5', 'r') as f:
        
                start = time.time()
    
                dhisto = f['histograms']
                X_sub = dhisto[:, 1, :]
                dparams = f['parameters']            
                y_sub = dparams[:, 0]
    
                end = time.time()
                print("si = ", si, " ri_in = ", ri_in, " Running time p1 = ", end - start)
                
                start = time.time()
                if ri_in == ri_out * rank_in_max:
                    X = X_sub
                    y = y_sub
                else:
                    X = np.concatenate((X, X_sub[:]), axis=0)
                    y = np.concatenate((y, y_sub[:]), axis=0)
    
                end = time.time()
                print("Running time p2 = ", end - start)

        if ri_out == 0:
            X_all = X
            y_all = y
        else:
            X_all = np.concatenate((X_all, X[:]), axis=0)
            y_all = np.concatenate((y_all, y[:]), axis=0)


    print('Shape of X_all read: ', X_all.shape)
    print('Shape of y_all read: ', y_all.shape)
    
    scaler = MinMaxScaler(copy=True)
    # normalize data
    
    X_post = scaler.fit_transform(X_all.T)
    X_post = X_post.T
    print('Shape of X_post: ', X_post.shape)

    with open(filename_out + '-Y.npy', 'wb') as f:
        np.save(f, X_post)
    with open(filename_out + '-P.npy', 'wb') as f:
        np.save(f, y_all)


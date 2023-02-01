from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import h5py
import time

start = time.time()

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
                     'trigonal',
                     'tetragonal']
#rank_max_list = [16, 16, 16, 16]
rank_max_list = [64, 512, 512, 512]

rank_in_max = 64
y_all = []
X_all = []

for si in range(4):

    rank_max = rank_max_list[si]
    path_in = path_in_list[si]
    filename_in = filename_in_list[si]

    for ri_out in range(0, int(rank_max / rank_in_max)):

        for ri_in in range(ri_out * rank_in_max, (ri_out + 1) * rank_in_max):
            
            with h5py.File(root_path + path_in + filename_in + str(ri_in) + '.hdf5', 'r') as f:
        
                dhisto = f['histograms']
                X_sub = dhisto[:, 1, :]
                dparams = f['parameters']            
                y_sub = dparams[:]
    
                if ri_in == ri_out * rank_in_max:
                    X = X_sub
                    y = y_sub
                else:
                    X = np.concatenate((X, X_sub[:]), axis=0)
                    y = np.concatenate((y, y_sub[:]), axis=0)
    
        if ri_out == 0:
            X_all.append(X)
            y_all.append(y)
        else:
            X_all[si] = np.concatenate((X_all[si], X[:]), axis=0)
            y_all[si] = np.concatenate((y_all[si], y[:]), axis=0)


    print('si = ', si, ' Shape of X_all read: ', X_all[si].shape)
    print('si = ', si, ' Shape of y_all read: ', y_all[si].shape)


X_all[1] = np.concatenate((X_all[1], X_all[2]), axis=0)
X_all.pop(2)
y_all[1] = np.concatenate((y_all[1], y_all[2]), axis=0)
y_all.pop(2)

# 0 for cubic,      [a, a, 1.5708, 0, 0, 1] 
# 1 for trigonal    [a, a, gamma,  1, 0, 0]
# 2 for tetragonal  [a, b, 1.5708, 0, 1, 0]

for si in range(3):

    filename_out = filename_out_list[si]
    scaler = MinMaxScaler(copy=True)
    # normalize data
    
    X_all[si] = scaler.fit_transform(X_all[si].T)
    X_all[si] = X_all[si].T
    print('si = ', si, ' Shape of X_all read: ', X_all[si].shape)
    print('si = ', si, ' Shape of y_all read: ', y_all[si].shape)

    with open(filename_out + '-Y.npy', 'wb') as f:
        np.save(f, X_all[si])
    with open(filename_out + '-P.npy', 'wb') as f:
        np.save(f, y_all[si])

end = time.time()
print("Total running time = ", end - start)


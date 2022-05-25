import os
import numpy as np

sets = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub', 'kinetics'
datasets = {
    'NTU/xview', 'NTU/xsub'
}

parts = {
    'joint', 'bone'
}

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            data_jpt = np.load('/data/ChenZhan/dataset/{}/{}_data_{}.npy'.format(dataset, set, part))            # train/val_data_joint/bone.py
            data_motion = np.load('/data/ChenZhan/dataset/{}/{}_data_{}_motion.npy'.format(dataset, set, part))  # train/val_data_joint/bone_motion.py
            N, C, T, V, M = data_jpt.shape
            data_jpt_motion = np.concatenate((data_jpt, data_motion), axis=1)
            np.save('/data/ChenZhan/dataset/{}/{}_data_{}_jm.npy'.format(dataset, set, part), data_jpt_motion)   # train/val_data_joint/bone_jm.py 

# -*- coding: utf-8 -*-

import nibabel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_path = r'E:\SZU\fmri机器学习\parc_data\data_load_data\nii\FunImgARCW\sub001\test.nii'
fMRI_data = nibabel.load(data_path)
fMRI_data = fMRI_data.get_fdata()
print(fMRI_data.shape)


fMRI_3RT = fMRI_data[:,:,:,2]

fMRI_3RT_x = fMRI_data[19,:,:,2]

plt.imshow(fMRI_3RT_x, 'gray')

fMRI_3RT_x = np.reshape(fMRI_data, [64,64,64])

mat_path = r'E:\SZU\fmri机器学习\parc_data\data_load_data\mat\mat_test.mat'
import scipy.io
mat_data = scipy.io.loadmat(mat_path)

mat_data = mat_data['data']


saved_path = r'E:\SZU\fmri机器学习\parc_data\data_load_data\mat\test.mat'
scipy.io.savemat(saved_path,
                 {"tmd": mat_data},
                 {"byd": [1,1,14,5,1,1]})

#首先导入np包
import numpy as np

#然后定义要加载的文件和路径
txt_path = r'E:\SZU\fmri机器学习\parc_data\data_load_data\txt\txt_test1.txt'

#之后我们使用刚刚导入的包加载数据
txt_data = np.loadtxt(txt_path)
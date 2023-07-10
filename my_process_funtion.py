
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from matplotlib import pylab as plt
from skimage import transform
from scipy.ndimage import zoom
import h5py
import os
from numpy import random
# import xlrd
# from sklearn.utils import shuffle
# from skimage.measure import compare_ssim

import pandas as pd

global nii_name

def check_image(data, step = 10,data_name = None,show_slices = True):
    width, height, queue = data.shape
    # show 3D image
    OrthoSlicer3D(data).show()

    if show_slices:
    # show per 2D imgage in 10 step
        x = int((queue/step)**0.5) + 1
        num = 1

        if data_name:
            plt.suptitle(['-Checking "'+data_name+'" in step '+str(step)], fontsize = 14)
        else:
            plt.suptitle(['-Checking "'+nii_name+'" in step '+str(step)], fontsize = 14)

        for i in range(0, queue, step):
            img_arr = data[:, :, i]
            plt.subplot(x, x, num)
            plt.imshow(img_arr, cmap='gray')
            num += 1

        plt.show()

def nii_loader(nii_path, check_img =  False):
    print('#Loading ', nii_path, '...')
    data = nib.load(nii_path)

    # print data msg
    # print(data)
    print("--Loading size:", data.shape)

    # check shape again
    # width, height, queue = data.dataobj.shape
    # print(width, height, queue)

    if(check_img):
        check_image(data)

    return data


# data resize
def my_resize(o_data, transform_size = None, transform_rate = None, check_img=False):
    print('#Resizing...')
    data = o_data
    print("--Original size:", data.shape)
    if transform_size:
        o_width, o_height, o_queue = data.shape
        width, height, queue = transform_size
        data = zoom(data, (width/o_width, height/o_height, queue/o_queue))
    elif transform_rate:
        data = zoom(data, transform_rate)
        # data = zoom(data, (transform_rate, transform_rate, transform_rate))

    print("--Transofmed size:", data.shape)

    if(check_img):
        check_image(data)

    return data


# window_change
def window_change(o_data, data_window = None, check_img=False):
    print('#Window changing...')
    data = o_data
    if data_window:
        voxel_min = data_window[0]
        voxel_max = data_window[1]
        data[data<voxel_min] = voxel_min
        data[data>voxel_max] = voxel_max

    if(check_img):
        check_image(data)

    return data


# voxel_dim_resampling   target_dim = [x,x,x] -1 for maintain
def voxel_dim_resampling(o_data, or_dim = None, target_dim = None, check_img=False):
    print('#Voxel dim resampling ...')
    data = o_data
    width, height, queue = data.shape
    if target_dim:
        dim1_rate = 1 if target_dim[0] == -1 else or_dim[0] / target_dim[0]
        dim2_rate = 1 if target_dim[1] == -1 else or_dim[1] / target_dim[1]
        dim3_rate = 1 if target_dim[2] == -1 else or_dim[2] / target_dim[2]
        # new_size = [int(dim1_rate*width), int(dim2_rate*height), int(dim3_rate*queue)]
        new_rate = [dim1_rate, dim2_rate, dim3_rate]

        # data = my_resize(data, transform_size=new_size)
        data = my_resize(data, transform_rate=new_rate)


    if(check_img):
        check_image(data)

    return data


# standardization
def standardizing(o_data, check_img=False):
    print('#Standardizing...')
    data = o_data
    width, height, queue = data.shape

    data_flatten = np.reshape(data, [1, -1])
    meann = np.mean(data_flatten)
    stdd = np.std(data_flatten)
    data = (data - meann) / stdd
    data = np.reshape(data, [width, height, queue])

    if(check_img):
        check_image(data)

    return data


# Linear_normalization
def linear_normalizing(o_data, check_img=False):
    print('#Linear_normalizing...')
    data = o_data
    minn = np.min(data)
    maxx = np.max(data)
    data = (data - minn) / (maxx - minn)

    if(check_img):
        check_image(data)

    return data


# centre window crop
def centre_window_cropping(o_data, reshapesize = None, check_img=False):
    print('#Centre window cropping...')
    data = o_data
    or_size = data.shape
    target_size = (reshapesize[0], reshapesize[1], or_size[2])

    # pad if or_size is smaller than target_size
    if (target_size[0] > or_size[0]) | (target_size[1] > or_size[1]) :
        if target_size[0] > or_size[0]:
            pad_size = int((target_size[0] - or_size[0]) / 2)
            data = np.pad(data, ((pad_size, pad_size),(0, 0), (0, 0)))
        if target_size[1] > or_size[1]:
            pad_size = int((target_size[1] - or_size[1]) / 2)
            data = np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)))

    #  centre_window_cropping
    cur_size = data.shape
    centre_x = float(cur_size[0] / 2)
    centre_y = float(cur_size[1] / 2)
    dx = float(target_size[0] / 2)
    dy = float(target_size[1] / 2)
    data = data[int(centre_x - dx + 1):int(centre_x + dx), int(centre_y - dy + 1): int(centre_y + dy), :]

    data = my_resize(data, transform_size=target_size)

    if(check_img):
        check_image(data)

    return data


# get the list for indices of arr (Rt.. arr > value)
def getListIndex(arr, value) :
    dim1_list = dim2_list = dim3_list = []
    if (arr.ndim == 3):
        index = np.argwhere(arr == value)
        dim1_list = index[:, 0].tolist()
        dim2_list = index[:, 1].tolist()
        dim3_list = index[:, 2].tolist()

    else :
        raise ValueError('The ndim of array must be 3!!')

    return dim1_list, dim2_list, dim3_list


# ROI cut
def ROI_cutting(o_data, o_roi, expend_voxel = 0, check_img=False):
    print('#ROI cutting...')
    data = o_data
    roi = o_roi

    [I1, I2, I3] = getListIndex(roi, 1)
    d1_min = min(I1)
    d1_max = max(I1)
    d2_min = min(I2)
    d2_max = max(I2)
    d3_min = min(I3)
    d3_max = max(I3)

    if expend_voxel > 0:
        d1_min -= expend_voxel
        d1_max += expend_voxel
        d2_min -= expend_voxel
        d2_max += expend_voxel
        # d3_min -= expend_voxel
        # d3_max += expend_voxel

        d1_min = d1_min if d1_min>0 else 0
        d1_max = d1_max if d1_max<data.shape[0]-1 else data.shape[0]-1
        d2_min = d2_min if d2_min>0 else 0
        d2_max = d2_max if d2_max<data.shape[1]-1 else data.shape[1]-1
        # d3_min = d3_min if d3_min>0 else 0
        # d3_max = d3_max if d3_max<data.shape[2]-1 else data.shape[2]-1

    data = data[d1_min:d1_max+1,d2_min:d2_max+1,d3_min:d3_max+1]
    roi = roi[d1_min:d1_max+1,d2_min:d2_max+1,d3_min:d3_max+1]
    # data = data[d1_min:d1_max+1,d2_min:d2_max+1,:]
    # roi = roi[d1_min:d1_max+1,d2_min:d2_max+1,:]

    print("--Cutting size:", data.shape)

    if(check_img):
        check_image(data)

    return data, roi


# block divide
def block_dividing(o_data, deep = None, step = None, adjust_num = 20,check_img=False):
    print('#Block dividing...')
    data = o_data
    data_group = []
    o_data_deep = data.shape[2]
    adjust_num = adjust_num

    if o_data_deep <= deep:
        tmp_data = np.zeros((data.shape[0], data.shape[1], deep))
        tmp_data[:, :, 0:o_data_deep] = data
        blocks = 1
        tmp_data = tmp_data
        data_group.append(tmp_data)

    else:
        blocks = (o_data_deep - deep) // step + 2
        if (o_data_deep - deep) % step == 0:
            blocks -= 1
        for i in range(blocks-1):
            tmp_data = data[:, :, (0 + i * step): (deep + i * step)]
            data_group.append(tmp_data)
        # tmp_data = np.zeros((data.shape[0],data.shape[1],deep))
        # tmp_data[:,:,0:(o_data_deep-(deep+i*step))] = data[:,:,(deep+i*step):o_data_deep]
        tmp_data = data[:, :, o_data_deep -deep:o_data_deep]
        data_group.append(tmp_data)

    # blocks_o = blocks
    # adjust_rate = 1
    # if blocks<adjust_num:
    #     adjust_rate = int(adjust_num/blocks)
    #     blocks = blocks*adjust_rate
    #     data_group = data_group*adjust_rate

    print("--Block size:", tmp_data.shape,
          " Divided number:(%d)"%(blocks))

    if(check_img):
        for i, divided_data in enumerate(data_group):
            divided_data_name = data_name + '_' + str(i)
            check_image(divided_data, step=1, data_name = divided_data_name)

    return data_group, blocks


# block divide
def block_dividing_with_gap(o_data, deep = None, step = None, adjust_num = 1,check_img=False):
    print('#Block dividing...')
    data = o_data
    data_group = []
    o_data_deep = data.shape[2]
    adjust_num = adjust_num

    if o_data_deep <= deep:
        tmp_data = np.zeros((data.shape[0], data.shape[1], deep))
        index_t = int((deep-o_data_deep)/2)
        tmp_data[:, :, index_t:index_t+o_data_deep] = data
        blocks = 1
        tmp_data = tmp_data
        data_group.append(tmp_data)

    else:
        blocks = o_data_deep//deep
        mod = o_data_deep%deep
        gap = blocks
        for i in range(blocks):
            index_t = list(range(i, o_data_deep-mod, gap))
            tmp_data = data[:, :, index_t]
            data_group.append(tmp_data)
        if (mod>0) :
            for i in range(mod):
                index_t = list(range(o_data_deep-i-1, mod-i-1, -gap))
                index_t = sorted(index_t)
                tmp_data = data[:, :, index_t]
                data_group.append(tmp_data)
                blocks = blocks+1


    blocks_o = blocks
    adjust_rate =1
    if blocks<adjust_num:
        adjust_rate = int(adjust_num/blocks)
        blocks = blocks*adjust_rate
        data_group = data_group*adjust_rate

    print("--Block size:", tmp_data.shape,
          " Divided number:%d(%d)"%(blocks, blocks_o),
          " Adjust rate:", adjust_rate)

    if(check_img):
        for i, divided_data in enumerate(data_group):
            divided_data_name = data_name + '_' + str(i)
            check_image(divided_data, step=1, data_name = divided_data_name)

    return data_group, blocks


# make h5 data
def make_h5_data(o_data, o_roi=None, label=None, h5_save_path=None,check_img=False):
    print('#Make h5 data...')
    data = o_data
    if (o_roi):
        roi = o_roi

    if (h5_save_path):
        for i, divided_data in enumerate(data):
            if not os.path.exists(os.path.join(h5_save_path, data_name)):
                os.makedirs(os.path.join(h5_save_path, data_name))
            save_file_name = os.path.join(h5_save_path, data_name, data_name + '_' + str(i+1) + '.h5')
            with h5py.File(save_file_name, 'a') as f:
                print("--h5 file path:", save_file_name,'    -label:', label, '    -size:', divided_data.shape)
                f['Data'] = divided_data
                f['Label'] = [label]
                if (o_roi):
                    f['ROI'] = roi[i]
    if(check_img):
        for i, divided_data in enumerate(data):
            divided_data_name = data_name + '_' + str(i)
            check_image(divided_data, data_name = divided_data_name)

# fold
def make_cross_fold(id_list, label_list, k):

    id_shuffle = id_list[:]
    random.shuffle(id_shuffle)
    fold_k = []
    group = int(len(label_list) / k)
    for i in range(k):
        if i == k - 1:
            fold_k.append(id_shuffle[i * group:])
        else:
            fold_k.append(id_shuffle[i * group:(i + 1) * group])
    print(fold_k)


# ==========================================================================
#  =========================    K-fold dived     ===========================
# K-fold dived
from sklearn.model_selection import StratifiedKFold

def  get_fold_filelist(seleted_data_list, csv_file, K=5, fold=1, random_state=2021, validation=False, validation_r = 0.2):
    """
    获取分折结果的API（基于size分3层的类别平衡分折）
    :param csv_file: 带有ID、LABEL的文件
    :param K: 分折折数
    :param fold: 返回第几折,从1开始
    :param random_state: 随机数种子
    :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
    :param validation_r: 抽取出验证集占训练集的比例
    :return: train和test的list，带有label和size
    """

    # load label list
    lable_list = np.loadtxt(open(csv_file, "rb"),delimiter=",", skiprows=0)
    lable_list = np.array(lable_list, dtype='int16')

    data_list = [i for i in lable_list if i[0] in seleted_data_list]
    data_list = np.array(data_list, dtype='int16')

    label = list(data_list[:,1])
    train_list = []
    test_list = []

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(label, label):
        train_list.append([data_list[i] for i in train])
        test_list.append([data_list[i] for i in test])

    if validation is False: # 不设置验证集，则直接返回
        train_set = train_list[fold-1]
        #train_set = shuffle(train_set)
        test_set = test_list[fold-1]
        #test_set = shuffle(test_set)
        return [train_set, test_set]
    else:
        train_p = [i for i in train_list[fold-1] if int(i[1]) == 1]
        train_n = [i for i in train_list[fold-1] if int(i[1]) == 0]

        validation_set = train_p[0:int(len(train_p) * validation_r)] + \
                         train_n[0:int(len(train_n) * validation_r)]
        train_set = train_p[int(len(train_p) * validation_r):] + \
                    train_n[int(len(train_n) * validation_r):]
        test_set = test_list[fold-1]

        return [train_set, validation_set, test_set]



if __name__ == '__main__':
    # ----------------------------Task 1 make data---------------------------------
    # Init
    data_phase = 'arterial'
    if data_phase == 'arterial':
        data_window = (-5, 145)
    elif data_phase == 'venous':
        data_window = (0, 200)
    else:
        raise ValueError('The data_phase of array must be arterial/venous!!')
    target_dim = (0.68, 0.68, -1) # -1 for maintain
    expend_voxel = int(20 // 0.68)  # 20//0.8
    reshapesize = (256, 256)
    deep = 16
    step = 8

    # path set

    data_root = "/home/szu/liver/final_nii_data"
    or_data_root = os.path.join(data_root,'extra_data/extra_data_all/'+data_phase+'_nii_data')
    #or_data_root = os.path.join(data_root, data_phase)
    save_root = '/home/szu/liver/final_nii_data'
    h5_save_path = os.path.join(save_root, 'arterial_extra_data_h5')

    # load label list
    lable_list = np.loadtxt(open(os.path.join(data_root, data_phase+'_new'+'.csv'), "rb"),delimiter=",", skiprows=0)
    lable_list = np.array(lable_list, dtype='float32')

    data_all_list = os.listdir(or_data_root)
    data_all_list.sort(key = lambda x:int(x[:-4]))

    for filename in data_all_list:
        data_name = filename.split('.')[0]# 0~999:nii_data   1000~1999:mask_data  3000~3999:segROI_data
        if int(data_name) > 999:
            continue

        #original CT data
        example_data_path = os.path.join(or_data_root, filename)
        print("=============Processing ", example_data_path, "=============")
        data = nii_loader(example_data_path, check_img=False)
        #data_name = data.dataobj.file_like.split('/')[-1].split('.')[0]
        or_dim = (np.abs(data.affine[0, 0]), np.abs(data.affine[1, 1]), np.abs(data.affine[2, 2]))
        label = lable_list[lable_list[:, 0] == int(data_name), 1]
        if int(data_name) <100 or (int(data_name) >500 and int(data_name) < 600 ):
            expend_voxel = 0
        else:
            expend_voxel = int(20 // or_dim[0])  # 20//0.68
        nii_name = data_name
        img_arr = np.array(data.dataobj, dtype='float32')

        # ROI data
        example_roi_path = os.path.join(or_data_root, str(int(data_name) + 1000) + '.nii')
        roi_data = nii_loader(example_roi_path, check_img=False)
        roi_arr = np.array(roi_data.dataobj, dtype='float32')
        roi_arr[roi_arr < 0.5] = 0
        roi_arr[roi_arr >= 0.5] = 1

        # # select images with ROI
        # img_arr = ROI_selecting(img_arr, roi_arr, check_img = False)
        # roi_arr = ROI_selecting(roi_arr, roi_arr, check_img=False)
        img_arr, roi_arr = ROI_cutting(img_arr, roi_arr, expend_voxel=expend_voxel, check_img=False)

        img_arr = window_change(img_arr, data_window=data_window, check_img=False)
        img_arr = voxel_dim_resampling(img_arr, or_dim = or_dim, target_dim = target_dim, check_img=False)
        img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize, check_img=False)
        img_arr = linear_normalizing(img_arr, check_img=False)

        img_arr = block_dividing(img_arr, deep=deep, step=step, check_img=False)
        # make h5 data
        make_h5_data(img_arr[0], label=label, h5_save_path=h5_save_path, check_img=False) # TODO:should be img_arr[0] don't konwn why

    print('Finish!')



    # ----------------------------Task 2 cal Similarity---------------------------------
    # Init
    '''data_phase = 'venous'
    if data_phase == 'arterial':
        data_window = (-5, 145)
    elif data_phase == 'venous':
        data_window = (0, 200)
    else:
        raise ValueError('The data_phase of array must be arterial/venous!!')
    target_dim = (0.68, 0.68, -1) # -1 for maintain
    # expend_voxel = int(20 // 0.68)  # 20//0.8
    reshapesize = (256, 256)

    data_root = "/media/szu/5d8781f1-6f57-42e7-a39e-d6c3966f7e61/wds/liver_diagnosis/final_nii_data"
    data1_root = os.path.join(data_root, 'af12' + data_phase)
    # data2_root = os.path.join(data_root, 'af12' + data_phase)
    data2_root = os.path.join(data_root,'extra_data/extra_data_all/'+data_phase+'_nii_data')

    # path set
    data_root = "/media/szu/5d8781f1-6f57-42e7-a39e-d6c3966f7e61/wds/HCC_RNA/data"
    data1_root = os.path.join(data_root, 'nii_gz_data', data_phase)
    data2_root = data1_root

    data1_list = os.listdir(os.path.join(data1_root))
    data2_list = os.listdir(os.path.join(data2_root))

    # load label list
    msg_data = []
    xl = xlrd.open_workbook(os.path.join(data_root, 'data_message.xlsx'))
    table = xl.sheet_by_name('Sheet1')
    nrows = table.nrows
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        if row:
            msg_data.append(row)
    msg_data = np.array(msg_data, dtype='float32')

    record = np.zeros((1, 3))
    for i,filename1 in enumerate(data1_list):
        example_data1_path = os.path.join(data1_root, filename1)
        print("=============Processing ", example_data1_path, "=============")
        data1 = nii_loader(example_data1_path, check_img=False)
        data1_name = data1.dataobj.file_like.split('/')[-1].split('.')[0]
        if int(data1_name) > 999:
            continue
        img1_arr = np.array(data1.dataobj, dtype='float32')

        case_index = msg_data[:, 0] == int(data1_name)
        or_dim = (np.abs(data1.affine[0, 0]), np.abs(data1.affine[1, 1]), np.abs(data1.affine[2, 2]))
        bottom, mid, top, x1, y1, x2, y2 = msg_data[case_index, 2:9].astype('int16').flatten()
        location = [x1,x2,y1,y2,img1_arr.shape[2]-top,img1_arr.shape[2]-bottom]
        expend_voxel = int(20 // or_dim[0])

        img1_arr = ROI_cutting(img1_arr, location, expend_voxel=expend_voxel, check_img=False)
        img1_arr = window_change(img1_arr, data_window=data_window, check_img=False)
        img1_arr = voxel_dim_resampling(img1_arr, or_dim=or_dim, target_dim=target_dim, check_img=False)
        img1_arr = linear_normalizing(img1_arr, check_img=False)
        # img1_arr = centre_window_cropping(img1_arr, reshapesize=reshapesize, check_img=False)
        # img1_arr = my_resize(img1_arr, transform_size=(reshapesize[0], reshapesize[1], img1_arr.shape[2]))


        for filename2 in data2_list[i:]:
            example_data2_path = os.path.join(data2_root, filename2)
            data2 = nii_loader(example_data2_path, check_img=False)
            nii_name = data2_name = data2.dataobj.file_like.split('/')[-1].split('.')[0]
            if int(data2_name) > 999:
                continue
            img2_arr = np.array(data2.dataobj, dtype='float32')

            case_index = msg_data[:, 0] == int(data2_name)
            or_dim = (np.abs(data2.affine[0, 0]), np.abs(data2.affine[1, 1]), np.abs(data2.affine[2, 2]))
            bottom, mid, top, x1, y1, x2, y2 = msg_data[case_index, 2:9].astype('int16').flatten()
            location = [x1, x2, y1, y2, img2_arr.shape[2] - top, img2_arr.shape[2] - bottom]
            expend_voxel = int(20 // or_dim[0])

            img2_arr = ROI_cutting(img2_arr, location, expend_voxel=expend_voxel, check_img=False)
            img2_arr = window_change(img2_arr, data_window=data_window, check_img=False)
            img2_arr = voxel_dim_resampling(img2_arr, or_dim=or_dim, target_dim=target_dim, check_img=False)
            img2_arr = linear_normalizing(img2_arr, check_img=False)
            # img2_arr = centre_window_cropping(img2_arr, reshapesize=reshapesize, check_img=False)
            img2_arr = my_resize(img2_arr, transform_size=img1_arr.shape,check_img=False)


            sore =  compare_ssim(img1_arr, img2_arr)

            record = np.vstack((record, np.array([int(data1_name), int(data2_name), sore])))
            excel_save_path = os.path.join( 'ssim_record2.xlsx')
            writer = pd.ExcelWriter(excel_save_path)
            result_gap = pd.DataFrame(record)
            result_gap.to_excel(writer, 'valid', float_format='%.4f')
            writer.save()
            writer.close()

            print("%s & %s ssim: %0.4f  %0.4f"%(data1_name, data2_name, sore, sore))'''

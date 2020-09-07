import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)

source_dir = 'data/UCF-101'
target_train_dir = 'data/UCF-101/train'
target_test_dir = 'data/UCF-101/test'
target_val_dir = 'data/UCF-101/val'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

train_txt_dir_list = []
with open('work/configs/trainlist01.txt', 'r') as f:
    txt = f.readlines()
    for i in txt:
        txt_dir, txt_class = i.split(' ')
        txt_dir1, txt_dir2 = txt_dir.split('/')
        train_txt_dir_list.append(txt_dir2[:-4])

test_txt_dir_list = []
with open('work/configs/testlist01.txt', 'r') as f:
    txt = f.readlines()
    for i in txt:
        txt_dir1, txt_dir2 = i.split('/')
        test_txt_dir_list.append(txt_dir2[:-5])
    
for key in label_dic:
    # 种类处理
    each_mulu = key + '_jpg'
    print(each_mulu, key)
    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)
    sum = len(label_mulu)
    for each_label_mulu in label_mulu:
        # 种类内部每个视频处理
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        #print(image_file[0])
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        frame = []
        vid = image_name
        for i in range(image_num):
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i+1) + '.jpg')
            frame.append(image_path)

        output_pkl1 = vid + '.pkl'
        #print(output_pkl1)
        for i in train_txt_dir_list:
            if vid==i:
                output_pkl = os.path.join(target_train_dir, output_pkl1)
                f = open(output_pkl, 'wb')
                pickle.dump((vid, label_dic[key], frame), f, -1)
                f.close()
                output_pkl = os.path.join(target_val_dir, output_pkl1)
                f = open(output_pkl, 'wb')
                pickle.dump((vid, label_dic[key], frame), f, -1)
                f.close()
        for i in test_txt_dir_list:
            if vid==i:
                output_pkl = os.path.join(target_test_dir, output_pkl1)
                f = open(output_pkl, 'wb')
                pickle.dump((vid, label_dic[key], frame), f, -1)
                f.close()

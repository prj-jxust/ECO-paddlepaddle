import os
import numpy as np
import cv2

video_src_src_path = 'data/UCF-101'
label_name = os.listdir(video_src_src_path)
label_dir = {}
txt_dir_list = {}
with open('work/configs/classInd.txt', 'r') as f:
    txt = f.readlines()
    for i in txt:
        txt_class, txt_name = i.split(' ')      
        txt_dir_list[txt_name[:-1]]=int(txt_class)-1
print(txt_dir_list)

for i in label_name:
    if i.startswith('.'):
        continue
    print(i)
    label_dir[i] = txt_dir_list[i]

    video_src_path = os.path.join(video_src_src_path, i)
    video_save_path = os.path.join(video_src_src_path, i) + '_jpg'
    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)

    videos = os.listdir(video_src_path)
    # 过滤出avi文件
    videos = filter(lambda x: x.endswith('avi'), videos)

    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        if not os.path.exists(video_save_path + '/' + each_video_name):
            os.mkdir(video_save_path + '/' + each_video_name)

        each_video_save_full_path = os.path.join(video_save_path, each_video_name) + '/'

        each_video_full_path = os.path.join(video_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        frame_count = 1
        success = True
        while success:
            success, frame = cap.read()
            # print('read a new frame:', success)

            params = []
            params.append(1)
            if success:
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)

            frame_count += 1
        cap.release()
np.save('label_dir.npy', label_dir)
print(label_dir)

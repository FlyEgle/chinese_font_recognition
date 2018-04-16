# -*- coding: utf-8 -*-
"""
从数据集中划分train和validation两个文件
train_test_split_ratio=0.1 or 0.2
Tree目录：
    data：
        train：
            folder1
            ......
            folder529
        validation:
            folder1
            ......
            folder529
"""
import os
import random
import PIL.Image as Image


# 检查路径下面是否都是文件
def isfile(path):
    for folder in os.listdir(path):
        if not os.path.isdir(path+folder):
            os.remove(path+folder)


# 建立文件夹
def mkdir(path):
    """
    if folder is exists, or make new dir
    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        print(path)
        print('success')
        return True
    else:
        print(path)
        print('folder is exist')
        return False


# 返回文件列表
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    child_file_name = []
    full_child_file_list = []
    for allDir in pathDir:
        if not allDir == '.DS_Store':
            child = os.path.join(filepath, allDir)

            full_child_file_list.append(child)
            child_file_name.append(allDir)
    return full_child_file_list, child_file_name


# 转移ratio文件
def move_ratio(data_list, original_str, replace_str):
    for x in data_list:
        fromImage = Image.open(x)
        x = x.replace(original_str, replace_str)
        fromImage.save(x)


if __name__ == '__main__':

    face_path = '/Users/jmc/Desktop/facepaper/newFaceScrub_clean/'
    data_tra_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra/'
    data_val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'

    full_child_file, child_file = eachFile(face_path)

    # 建立相应的文件夹
    for i in child_file:
        tra_path = data_tra_path + '/' + str(i)
        mkdir(tra_path)
        val_path = data_val_path + '/' + str(i)
        mkdir(val_path)

    # 划分train和val
    test_train_split_ratio = 0.9

    for i in full_child_file:
        pic_dir, pic_name = eachFile(i)
        random.shuffle(pic_dir)
        train_list = pic_dir[0:int(test_train_split_ratio * len(pic_dir))]
        val_list = pic_dir[int(test_train_split_ratio * len(pic_dir)):]

        # train_move, val_move
        print('proprecessing %s' % i)

        move_ratio(train_list, 'newFaceScrub_clean', 'face_data/face_tra')
        move_ratio(val_list, 'newFaceScrub_clean', 'face_data/face_val')



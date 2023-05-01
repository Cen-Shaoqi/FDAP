from utils import *
import os
from shutil import copy

setup_seed(42)


is_random = True
# site = 'CNG'
# folder = 'CNG_val_shuffle'
month = 1
floor = 5
rootDir = f'/nfs/UJI_LIB/data/true_radioMap/floor_{floor}/month_{month}/'
saveDir = '/nfs/UJI_LIB/data/updateDataset'
folder = f'floor_{floor}'
path = rootDir

if not os.path.isdir(path):
    os.makedirs(path)

path_ls = ls(path)

if is_random:
    random.shuffle(path_ls)
print(path_ls)

datasetLen = len(path_ls)

# 按文件名顺序以 8：2 的比例划分 train_set 和 test_set
# splitRadio = 0.8
# train_ls = path_ls[: int(datasetLen * splitRadio)]
# test_ls = path_ls[int(datasetLen * splitRadio):]

train_radio = 0.6
val_radio = 0.2
test_radio = 1 - train_radio - val_radio

train_ls = path_ls[:int(datasetLen * train_radio)]
val_ls = path_ls[int(datasetLen * train_radio): int(datasetLen * (val_radio + train_radio))]
test_ls = path_ls[int(datasetLen * (1-test_radio)):]

# print(len(train_ls))
# print(len(val_ls))
# print(len(test_ls))

for file in train_ls:
    from_path = oj(rootDir, file)
    to_path = oj(saveDir, folder, 'train')

    if not os.path.isdir(to_path):
        os.makedirs(to_path)
    copy(from_path, to_path)

for file in val_ls:
    from_path = oj(rootDir, file)
    to_path = oj(saveDir, folder, 'val')

    if not os.path.isdir(to_path):
        os.makedirs(to_path)
    copy(from_path, to_path)


for file in test_ls:
    from_path = oj(rootDir, file)
    to_path = oj(saveDir, folder, 'test')

    if not os.path.isdir(to_path):
        os.makedirs(to_path)
    copy(from_path, to_path)



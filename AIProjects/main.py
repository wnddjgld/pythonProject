import os
from PIL import Image
import numpy as np

all_files = []
for i in range(0, 10):
    path_dir = './images/training/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)

x_train_datas = []
y_train_datas = []

for num in range(0, 10):
    for numbers in all_files[num]:
        img_path = './images/training/{0}/{1}'.format(num, numbers)
        print("lord: " + img_path)
        img = Image.open(img_path)

        imgarr = np.array(img) / 255.0
        x_train_datas.append(np.reshape(imgarr, newshape=(784, 1)))  # 입력값
        y_tab = np.zeros(shape=(10))
        y_tab[num] = 1
        y_train_datas.append(y_tab)  # 정답

print(len(x_train_datas))
print(len(y_train_datas))

# -===================
eval_files = []
for i in range(0, 10):
    path_dir = './images/testing/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)

x_test_datas = []
y_test_datas = []

for num in range(0, 10):
    for numbers in eval_files[num]:
        img_path = './images/testing/{0}/{1}'.format(num, numbers)
        print("lord: " + img_path)
        img = Image.open(img_path)
        imgarr = np.array(img) / 255.0
        x_test_datas.append(np.reshape(imgarr, newshape=(784, 1)))  # 입력값
        y_tab = np.zeros(shape=(10))
        y_tab[num] = 1
        y_test_datas.append(y_tab)  # 정답

print(len(x_test_datas))
print(len(y_test_datas))

x_train_datas = np.reshape(x_train_datas, newshape=(-1, 784))
y_train_datas = np.reshape(y_train_datas, newshape=(-1, 10))


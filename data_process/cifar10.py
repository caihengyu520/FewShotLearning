from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


# def load_data():
#     base = "file:///D:/fashionmnist/"
#     files = [
#         'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
#         't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
#     ]
#
#     paths = []
#     for fname in files:
#         paths.append(get_file(fname, origin=base+fname))
#
#     with gzip.open(paths[0], 'rb') as lbpath:
#         y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
#
#     with gzip.open(paths[1], 'rb') as imgpath:
#         x_train = np.frombuffer(
#             imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
#
#     with gzip.open(paths[2], 'rb') as lbpath:
#         y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
#
#     with gzip.open(paths[3], 'rb') as imgpath:
#         x_test = np.frombuffer(
#             imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
#
#     return (x_train, y_train), (x_test, y_test)
#
#
# print(tf.__version__)
# (train_images, train_labels), (test_images, test_labels) = load_data()
#
# # for i in [0,1,2,3,4,5,6,7,8,9]:
# #     train_images_1=train_images.reshape([60000,28*28])
# #     data=pd.DataFrame(train_images_1)
# #     label=pd.Series(train_labels)
# #     list_1=list(label[label==i].index)
# #     # print(data)
# #     # print(list_1)
# #     # print(data.iloc[list_1])
# #     data_file='fashion_mnist/train/leibie_'+str(i)
# #     data.iloc[list_1].to_csv(data_file,index=False)
# #
# #     # data_1=data[label,:]
# #     # print(data_1)

test_data=pd.read_csv('cifar-10-batches-py/test.csv')
print(test_data)
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    list_1 = test_data.index[test_data['3072'] == i].tolist()
    # print(data)
    # print(list_1)
    # print(data.iloc[list_1])
    # data_file = 'cifar10_class_data/test/leibie_' + str(i)+'.csv'
    # test_data.iloc[list_1].to_csv(data_file, index=False)
    print(test_data.iloc[list_1])

    # data_1=data[label,:]
    # print(data_1)

train_data=pd.read_csv('cifar-10-batches-py/train.csv')
print(train_data)
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    list_1 = train_data.index[train_data['3072'] == i].tolist()
    # print(data)
    # print(list_1)
    # print(data.iloc[list_1])
    # data_file = 'cifar10_class_data/train/leibie_' + str(i)+'.csv'
    # train_data.iloc[list_1].to_csv(data_file, index=False)
    print(train_data.iloc[list_1])

    # data_1=data[label,:]
    # print(data_1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

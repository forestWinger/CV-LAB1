# 下载并准备数据集Fashion-MNIST

import os
import gzip
import numpy as np
import urllib.request

def load_fashion_mnist(path, kind='train'):
    """Load Fashion-MNIST data from `path`"""
    base_url = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/'

    if kind == 'train':
        labels_url = base_url + 'train-labels-idx1-ubyte.gz'
        images_url = base_url + 'train-images-idx3-ubyte.gz'
    elif kind == 't10k':
        labels_url = base_url + 't10k-labels-idx1-ubyte.gz'
        images_url = base_url + 't10k-images-idx3-ubyte.gz'
    else:
        raise ValueError("Invalid 'kind' parameter. Use 'train' or 't10k'.")

    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, path)

    # 创建文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    labels_path = os.path.join(save_path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(save_path, '%s-images-idx3-ubyte.gz' % kind)

    # 下载标签文件
    if not os.path.exists(labels_path):
        urllib.request.urlretrieve(labels_url, labels_path)

    # 下载图像文件
    if not os.path.exists(images_path):
        urllib.request.urlretrieve(images_url, images_path)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
        # 使用One-Hot编码
        new_labels = np.zeros((len(labels), 10), dtype=np.uint8)
        for i, label1 in enumerate(labels):
            new_labels[i][label1] = 1


    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images/255, new_labels

X_train, y_train = load_fashion_mnist('data/fashion', kind='train')
X_test, y_test = load_fashion_mnist('data/fashion', kind='t10k')
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)





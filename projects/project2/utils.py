import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

FILE_PATH = "."
TRAIN_FEATURES_PATH = FILE_PATH + '/data/train/images'
TRAIN_LABELS_PATH = FILE_PATH + '/data/train/groundtruth'
TRAIN_SAMPLES = 100
TEST_FEATURES_PATH = FILE_PATH + '/data/test_set_images'
TEST_SAMPLES = 50
VALIDATION_RATIO = 0.2
USE_SHUFFLE = False


def load_train_data():
    x_train = []
    y_train = []
    for i in range(TRAIN_SAMPLES):
        img_name = '/satImage_{:03}.png'.format(i+1)
        train_feature = plt.imread(TRAIN_FEATURES_PATH + img_name)
        train_label = plt.imread(TRAIN_LABELS_PATH + img_name)
        x_train.append(train_feature)
        y_train.append(train_label)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    return x_train, y_train


def load_test_data():
    data = []
    for i in range(TEST_SAMPLES):
        img_name = '/test_{}/test_{}.png'.format(i+1, i+1)
        test_feature = plt.imread(TEST_FEATURES_PATH + img_name)
        data.append(test_feature)
    data = np.asarray(data)
    return data


def transform_labels(y_train):
    y_train = np.round(y_train)
    y_train = y_train[..., np.newaxis]
    return y_train


def train_test_split(x, y, validation_ratio=VALIDATION_RATIO, shuffle=USE_SHUFFLE):
    train_num = int(np.round(x.shape[0] * (1 - validation_ratio)))
    if shuffle:
        indexes = np.random.permutation(np.arange(x.shape[0]))
        x = x[indexes]
        y = y[indexes]
    x_test = x[train_num:,:,:,:]
    y_test = y[train_num:,...]
    x_train = x[:train_num,...]
    y_train = y[:train_num,...]
    return x_train, y_train, x_test, y_test


def save_preds(preds):
    path = '/data/predictions'
    for i in range(len(preds)):
        img = Image.fromarray((preds[i,:,:,0] * 255.).astype(np.uint8))
        full_name = FILE_PATH + path + f'/pred_{i+1}.png'
        img.save(full_name)

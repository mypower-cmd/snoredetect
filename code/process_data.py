import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler   #实现z-score标准化
from config import cfg

# MFCC
# train_data_dir = "D:/Project/snore/Snore_dataset_MFCC/train/"
# val_data_dir = "D:/Project/snore/Snore_dataset_MFCC/val/"
# test_data_dir = "D:/Project/snore/Snore_dataset_MFCC/test/"
# PLP
train_data_dir = "D:/Project/snore/Snore_dataset_PLP/train/"
val_data_dir = "D:/Project/snore/Snore_dataset_PLP/val/"
test_data_dir = "D:/Project/snore/Snore_dataset_PLP/test/"
# LPCC
# train_data_dir = "D:/Project/snore/Snore_dataset_LPCC/train/"
# val_data_dir = "D:/Project/snore/Snore_dataset_LPCC/val/"
# test_data_dir = "D:/Project/snore/Snore_dataset_LPCC/test/"


label_train_txt = "./data_list/label_train.txt"
label_val_txt = "./data_list/label_val.txt"
label_test_txt = "./data_list/label_test.txt"

shuffle_label_train_txt = "./data_list/shuffle_label_train.txt"
shuffle_label_val_txt = "./data_list/shuffle_label_val.txt"

tfrecord_train = "./tfrecords/train.tfrecords"
tfrecord_val = "./tfrecords/val.tfrecords"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value))  # 本身就给得是数组，这里不加[]


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def gen_label_file(data_base_dir, file_label_save_dir):
    """
    生成数据标签列表文件
    :param data_base_dir: 数据所在基础目录
    :param file_label_save_dir: 生成的数据标签文件保存全路径
    :return:
    """
    curr_train_dirs = os.listdir(data_base_dir)
    with open(file_label_save_dir, "w") as f:
        for index, curr_train_dir in enumerate(curr_train_dirs):
            pwd_train_dir = data_base_dir + curr_train_dir
            csv_file_names = os.listdir(pwd_train_dir)
            for _, csv_file_name in enumerate(csv_file_names):
                pwd_csv_file = pwd_train_dir + "/" + csv_file_name
                file_and_label = pwd_csv_file + "\t" + str(index) + "\n"
                f.write(file_and_label)
    f.close()
    print("gen finished")


def shuffle_label_file(in_file, out_file):
    """
    随机扰乱标签文件
    :param in_file: 未被扰乱顺序的数据标签文件
    :param out_file: 扰乱后的数据标签文件
    :return:
    """
    lines = []
    with open(out_file, 'w') as out:
        with open(in_file, 'r') as infile:
            for line in infile:
                lines.append(line)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        for line in lines:
            out.write(line)
        infile.close()
    out.close()
    print("shuffle file finished")


def get_data_labels(label_file):
    """
    从数据标签文件列表文件中读取文件名和标签
    :param label_file: 数据标签文件
    :return:文件名，标签
    """
    filenames = []
    labels = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        # 抽小批量测试
        # random.seed(42)
        # lines_test = random.sample(lines, 1800)
        # for line in lines_test:
        for line in lines:
            filenames.append(line.split()[0])
            labels.append(line.split()[1])
    f.close()
    return filenames, labels


def gennerate_terecord_file(tfrecordfilename, label_file):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :param label_file: 之前生成的数据标签文件
    :return:
    """
    cnt = 0
    filenames, labels = get_data_labels(label_file)
    with tf.io.TFRecordWriter(tfrecordfilename) as writer:
        for filename, label in zip(filenames, labels):
            cnt = cnt + 1
            csv_data = pd.read_csv(filename, header=None, index_col=0).values[1:]
            csv_data = np.float32(
                csv_data.flatten())  # 别直接用float, 因为numpy和python互不相关
            label = int(label)
            feature = {
                'data': _float_feature(csv_data),
                'label': _int64_feature(label)
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            if cnt % 100 == 0:
                print("the length of tfrecords is: %d " % cnt)
        print("total : %d " % cnt)


def _parse_example(example_string):
    """
    tfrecod数据解析
    :param example_string:
    :return:
    """
    # 浮点型数组时不能用FixedLenFeature，要用FixedLenSequenceFeature！！！！！！！
    feature_description = {
        'MFCC': tf.io.FixedLenFeature([42*12], tf.float32),
        'LPCC': tf.io.FixedLenFeature([42*12], tf.float32),
        'PLP': tf.io.FixedLenFeature([42*12], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    feature_dict = tf.io.parse_example(example_string,
                                       feature_description)


    MFCCs = tf.reshape(feature_dict['MFCC'], [42, 12])
    LPCCs = tf.reshape(feature_dict['LPCC'], [42, 12])
    PLPs = tf.reshape(feature_dict['PLP'], [42, 12])
    MFCCs = tf.convert_to_tensor(MFCCs, dtype=tf.float32)
    LPCCs = tf.convert_to_tensor(LPCCs, dtype=tf.float32)
    PLPs  = tf.convert_to_tensor(PLPs, dtype=tf.float32)
    # std = StandardScaler()  # 训练数据，赋值给b_test
    # MFCCs = std.fit_transform(MFCCs)
    # LPCCs = std.fit_transform(LPCCs)
    # PLPs = std.fit_transform(PLPs)

    data = MFCCs + LPCCs + PLPs

    labels = feature_dict['label']
    labels = tf.cast(labels, tf.int64)

    return data, labels


def gen_data_batch(file_pattern, batch_size, num_repeat=1, is_training=True):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    if is_training:
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
def gen_valdata_batch(file_pattern, batch_size):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.repeat(1)
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_truelabel(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(map_func=_parse_example)
    truelabel = []
    for data,label in dataset:
        label = np.int64(label)
        truelabel.extend(label)
    return truelabel

if __name__ == '__main__':
    # gen_label_file(train_data_dir, label_train_txt)  # 训练集数据标签列表文件生成
    # gen_label_file(val_data_dir, label_val_txt)  # 验证集数据标签列表文件生成
    # gen_label_file(test_data_dir, label_test_txt)  # 测试集集数据标签列表文件生成
    # shuffle_label_file(label_train_txt, shuffle_label_train_txt)  # 对训练用数据标签文件顺序进行扰乱
    # gennerate_terecord_file(tfrecord_train, shuffle_label_train_txt)  # 训练数据tfrecord文件生成
    # gennerate_terecord_file(tfrecord_val, label_val_txt)  # 验证数据tfrecord文件生成
    print(">>>>>>>>>>")
    gen_data_batch(cfg.train_dataset_path, cfg.batch_size, 1, is_training=True)

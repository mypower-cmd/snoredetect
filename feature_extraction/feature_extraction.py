import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from spafe.features import lpc
from spafe.utils.preprocessing import SlidingWindow
import librosa.display
import python_speech_features as psf
from spafe.features import rplp
import tensorflow as tf
import sys
sys.path.append('../code/')
from config import cfg

# train_data_dir = "../dataset/train/"
# val_data_dir = "../dataset/val/"
# test_data_dir = "../dataset/test/"
#
# train_save_dir = "../tfrecords/"
# val_save_dir = "../tfrecords/"
# test_save_dir = "../tfrecords/"

data_dir = "../dataset/after_process/"
save_dir = "../tfrecords/snore_dataset.tfrecords"

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

def feature_extracion(data_base_dir, data_save_dir):
    curr_train_dirs = os.listdir(data_base_dir)
    all_num = 0
    index = 0
    for curr_train_dir in curr_train_dirs:
        pwd_train_dir = data_base_dir + curr_train_dir
        wav_file_names = os.listdir(pwd_train_dir)
        all_num += len(wav_file_names)
    print("all_num:", all_num)
    m = np.array(all_num)
    np.save(cfg.num_samples_file_name, m)  # 保存
    with tf.io.TFRecordWriter(data_save_dir) as trainwriter:
        for curr_train_dir in curr_train_dirs:
            print(curr_train_dir)
            # 鼾声和非鼾声数据集
            pwd_train_dir = data_base_dir + curr_train_dir
            # 鼾声数据名
            wav_file_names = os.listdir(pwd_train_dir)
            for wav_file_name in wav_file_names:
                pwd_wav_file = pwd_train_dir + "/" + wav_file_name
                # python_speech_features特征提取
                # 导入音频
                (rate, sig) = wav.read(pwd_wav_file)
                try:
                    # MFCC
                    mfccs = psf.base.mfcc(
                        sig,
                        samplerate=24000,
                        winlen=0.0853,
                        winstep=0.0853,
                        nfilt=40,
                        nfft=2048,
                        preemph=0.85,
                        winfunc=np.hanning)
                    # LPCC
                    lpccs = lpc.lpcc(sig,
                                     fs=24000,
                                     order=13,
                                     pre_emph=True,
                                     pre_emph_coeff=0.85,
                                     window=SlidingWindow(0.0853, 0.0853, "hanning"))
                    plps = rplp.plp(sig,
                                    fs=24000,
                                    order=14,
                                    pre_emph=True,
                                    pre_emph_coeff=0.85,
                                    window=SlidingWindow(0.0853, 0.0853, "hanning"),
                                    nfft=2048,
                                    fbanks=None)
                except np.linalg.LinAlgError:
                    print("np.linalg.LinAlgError:", pwd_wav_file)
                    continue
                else:
                    # 取2-13维数据
                    mfcc = mfccs[:, 1:]
                    # MFCC的数据不足补0
                    mfcc = np.array(mfcc)
                    mfcc = mfcc.T
                    new_rol = np.zeros((12, 1), dtype=float)
                    while len(mfcc[0]) < 42:
                        mfcc = np.append(mfcc, new_rol, axis=1)
                    mfcc = mfcc.T

                    # 取2-13维数据
                    lpccs = lpccs[:, 1:]
                    # LPCC的数据不足补0
                    lpccs = np.array(lpccs)
                    lpccs = lpccs.T
                    new_rol = np.zeros((12, 1), dtype=float)
                    while len(lpccs[0]) < 42:
                        lpccs = np.append(lpccs, new_rol, axis=1)
                    lpcc = lpccs.T

                    # 取2-13维数据
                    plps = plps[:, 2:]
                    # PLP的数据不足补0
                    plps = np.array(plps)
                    plps = plps.T
                    new_rol = np.zeros((12, 1), dtype=float)
                    while len(plps[0]) < 42:
                        plps = np.append(plps, new_rol, axis=1)
                    plp = plps.T

                    label = int(curr_train_dir)
                    print(label)
                    # print(np.shape(feature_data[2]))
                    feature = {
                        'MFCC': _float_feature(mfcc.flatten()),
                        'LPCC': _float_feature(lpcc.flatten()),
                        'PLP': _float_feature(plp.flatten()),
                        'label': _int64_feature(label)
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature))
                    trainwriter.write(example.SerializeToString())
                    index += 1
                    print("data process complete : " + str(index / all_num * 100) + '%')
        trainwriter.close()


if __name__ == '__main__':
    # transform(1, test_data_dir, test_save_dir)
    # transform(1, train_data_dir, train_save_dir)
    feature_extracion(data_dir, save_dir)

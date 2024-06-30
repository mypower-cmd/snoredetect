import os
import numpy as np
import pandas as pd
import librosa.display
import scipy.io.wavfile as wav
import python_speech_features as psf
# import matplotlib.pyplot as plt

train_data_dir = "D:/Project/snore/Aug_Snore/train/"
val_data_dir = "D:/Project/snore/Aug_Snore/val/"
test_data_dir = "D:/Project/snore/Aug_Snore/test/"

train_save_dir = "D:/Project/snore/Snore_dataset_MFCC/train/"
val_save_dir = "D:/Project/snore/Snore_dataset_MFCC/val/"
test_save_dir = "D:/Project/snore/Snore_dataset_MFCC/test/"


def transform(mode, data_base_dir, data_save_dir):
    curr_train_dirs = os.listdir(data_base_dir)
    for curr_train_dir in curr_train_dirs:
        # 鼾声和非鼾声数据集
        pwd_train_dir = data_base_dir + curr_train_dir
        # 鼾声数据名
        wav_file_names = os.listdir(pwd_train_dir)
        for wav_file_name in wav_file_names:
            pwd_wav_file = pwd_train_dir + "/" + wav_file_name
            if mode == 0:
                # librosa特征处理
                # 导入音频
                y, sr = librosa.load(pwd_wav_file,
                                     sr=24000,
                                     mono=True,
                                     offset=0.0)
                # 提取mfcc系数
                mfccs = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    S=None,
                    n_fft=2048,
                    hop_length=2048,
                    n_mfcc=13,
                    n_mels=40)
                # 取2-13维数据
                mfcc = mfccs[1:, :]
                # MFCC的数据不足补0
                mfcc = np.array(mfcc)
                new_rol = np.zeros((12, 1), dtype=float)
                while len(mfcc[0]) < 42:
                    mfcc = np.append(mfcc, new_rol, axis=1)
                # 矩阵转置
                mfcc = mfcc.T
            elif mode == 1:
                # python_speech_features特征提取
                # 导入音频
                (rate, sig) = wav.read(pwd_wav_file)
                # print(rate)
                # 提取mfcc系数
                mfccs = psf.base.mfcc(
                    sig,
                    samplerate=24000,
                    winlen=0.0853,
                    winstep=0.0853,
                    nfilt=40,
                    nfft=2048,
                    preemph=0.85,
                    winfunc=np.hanning)
                # 取2-13维数据
                mfcc = mfccs[:, 1:]
                # MFCC的数据不足补0
                mfcc = np.array(mfcc)
                mfcc = mfcc.T
                new_rol = np.zeros((12, 1), dtype=float)
                while len(mfcc[0]) < 42:
                    mfcc = np.append(mfcc, new_rol, axis=1)
                mfcc = mfcc.T
            # 保存成csv文件
            df = pd.DataFrame(mfcc)
            save_dir = data_save_dir + curr_train_dir + "/" + wav_file_name.split('.')[0]
            df.to_csv(save_dir + '.csv')
            print(wav_file_name.split('.')[0])
    print("Transition finished")


if __name__ == '__main__':
    # transform(1, test_data_dir, test_save_dir)
    # transform(1, train_data_dir, train_save_dir)
    transform(1, val_data_dir, val_save_dir)

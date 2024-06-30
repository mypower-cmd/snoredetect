import numpy as np
import librosa.display
import pandas as pd
import scipy.io.wavfile as wav
import python_speech_features as psf
from spafe.features import rplp
from spafe.features import lpc
from spafe.utils.preprocessing import SlidingWindow

wav_file = "new_snore_0000.wav"


def transform_test(mode):
    global mfcc
    if mode == 0:
        # librosa特征处理
        # 导入音频
        y, sr = librosa.load(
            wav_file,
            sr=24000, mono=True, offset=0.0)
        # 查看音频时长和采样率
        p = librosa.get_duration(
            path=wav_file)
        print(p, sr)
        # 提取mfcc系数
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            S=None,
            n_fft=2048,
            hop_length=2048,
            n_mfcc=13,
            n_mels=40)
        print(mfccs)
        print(len(mfccs), len(mfccs[0]))
        # 取2-13维数据
        mfccs = mfccs[1:, :]
        # MFCC的数据不足补0
        mfccs = np.array(mfccs)
        new_rol = np.zeros((12, 1), dtype=float)
        while len(mfccs[0]) < 42:
            mfccs = np.append(mfccs, new_rol, axis=1)
        # 矩阵转置
        mfccs = mfccs.T
        # 保存成csv文件
        df = pd.DataFrame(mfccs)
        df.to_csv('new_snore_0001.csv')
    elif mode == 1:
        # python_speech_features特征提取
        # 导入音频
        (rate, sig) = wav.read(wav_file)
        print(rate)

        # 提取mfcc系数
        mfccs = psf.base.mfcc(sig,
                              samplerate=24000,
                              numcep=13,
                              winlen=0.0853,
                              winstep=0.0853,
                              nfilt=40,
                              nfft=2048,
                              preemph=0.85,
                              winfunc=np.hanning)
        print(mfccs)
        print(len(mfccs), len(mfccs[0]))
        # 取2-13维数据
        mfccs = mfccs[:, 1:]
        # MFCC的数据不足补0
        mfccs = np.array(mfccs)
        mfccs = mfccs.T
        # print(mfccs)
        new_rol = np.zeros((12, 1), dtype=float)
        while len(mfccs[0]) < 42:
            mfccs = np.append(mfccs, new_rol, axis=1)
        mfccs = mfccs.T
        # 保存成csv文件
        df = pd.DataFrame(mfccs)
        df.to_csv('new_snore_0001.csv')
    elif mode == 2:
        # 读取音频文件
        rate, sig = wav.read(wav_file)
        # 提取PLP特征
        plps = rplp.plp(sig,
                        fs=24000,
                        order=13,
                        pre_emph=True,
                        pre_emph_coeff=0.85,
                        window=SlidingWindow(0.0853, 0.0853, "hanning"),
                        nfft=2048)
        # 输出PLP特征
        # print(plps)
        # print(len(plps), len(plps[0]))
        # # 取2-13维数据
        plps = plps[:, 1:]
        print(plps)
        print(len(plps), len(plps[0]))
        #
        # # PLP的数据不足补0
        # plps = np.array(plps)
        # plps = plps.T
        # print(plps)
        # new_rol = np.zeros((12, 1), dtype=float)
        # while len(plps[0]) < 42:
        #     plps = np.append(plps, new_rol, axis=1)
        # plps = plps.T
        # print(plps)
        # 保存成csv文件
        df = pd.DataFrame(plps)
        df.to_csv('new_snore_0000.csv')
        # b = pd.read_csv('new_snore_0000.csv', header=None, index_col=0).values[1:]
        # print(b)
        # print(len(b), len(b[0]))
    elif mode == 3:
        # 读取音频文件
        rate, sig = wav.read(wav_file)
        # 提取LPCC特征
        lpccs = lpc.lpcc(sig,
                         fs=24000,
                         order=13,
                         pre_emph=True,
                         pre_emph_coeff=0.85,
                         window=SlidingWindow(0.0853, 0.0853, "hanning"))
        # 输出LPCC特征
        print(lpccs)
        print(len(lpccs), len(lpccs[0]))
        # # 取2-13维数据
        # lpccs = lpccs[:, 1:]
        # # LPCC的数据不足补0
        # lpccs = np.array(lpccs)
        # lpccs = lpccs.T
        # new_rol = np.zeros((12, 1), dtype=float)
        # while len(lpccs[0]) < 42:
        #     lpccs = np.append(lpccs, new_rol, axis=1)
        # lpccs = lpccs.T
        # 保存成csv文件
        df = pd.DataFrame(lpccs)
        df.to_csv('new_snore_0000.csv')


if __name__ == '__main__':
    transform_test(2)

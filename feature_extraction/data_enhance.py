import librosa
from scipy.io import wavfile
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

input_path = r"D:\Project\snore\Aug_Snore\val"
output_path = r"D:\Project\snore\Aug_Snore\val\new"


def pitch_shifting(input_path, output_path, n_steps=2, bins_per_octave=24):
    """
    fs: 音频采样率
    n_steps: 要移动多少步
    bins_per_octave: 每个八度音阶(半音)多少步
    """
    for folder in range(0, 2):
        for i, file in enumerate(os.listdir(input_path + "\\" + str(folder))):
            input = input_path + "\\" + str(folder) + '\\' + file
            output = output_path + "\\" + str(folder) + '\\' + file.split(".")[0] + "_shift%05d.wav" % i
            #     for i, file in enumerate(os.listdir(input_path)):
            #         input = input_path + "\\" + file
            #         output = output_path + "\\" + file.split(".")[0] + "_shift%04d.wav" % i
            wav_data, fs = librosa.load(input, sr=24000, mono=True)
            wav_data = librosa.effects.pitch_shift(wav_data, sr=fs, n_steps=n_steps, bins_per_octave=bins_per_octave)
            wavfile.write(output, int(fs), wav_data)
            print(f"{file} is shifted.")


def add_noise(input_path, output_path):
    # for folder in os.listdir(input_path):
    for folder in range(0, 2):
        for i, file in enumerate(os.listdir(input_path + "\\" + str(folder))):
            file1 = input_path + "\\" + str(folder) + '\\' + file
            file2 = output_path + "\\" + str(folder) + '\\' + file.split(".")[0] + "_add%05d.wav" % i
            #     for i, file in enumerate(os.listdir(input_path)):
            #         file1 = input_path + "\\" + file
            #         file2 = output_path + "\\" + file.split(".")[0] + "_add%04d.wav" % i
            wav_data, fs = librosa.load(file1, sr=24000, mono=True)
            wn = np.random.randn(len(wav_data))
            # wav_data = np.where(wav_data != 0.0, wav_data + 0.005 * wn, 0.0)  # 噪声不要添加到0上
            wav_data = wav_data + 0.005 * wn
            wavfile.write(file2, int(fs), wav_data)
            print(f"{file} is added.")


def moveToTar(input_path, output_path):
    for folder in range(0, 2):
        for file in os.listdir(input_path + "\\" + str(folder)):
            file1 = input_path + "\\" + str(folder) + '\\' + file
            file2 = output_path + "\\" + str(folder) + '\\' + file
            shutil.move(file1, file2)
            print(f"{file} is moved.")


if __name__ == '__main__':
    # add_noise(input_path, output_path)
    # moveToTar(output_path, input_path)
    # pitch_shifting(output_path, input_path)
    moveToTar(output_path, input_path)

# --------------------------画图------------------------------
# wav_data1, _ = librosa.load("./new_snore_0000_add.wav", sr=fs, mono=True)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
# # ########### 画图
# plt.subplot(2, 2, 1)
# plt.title("语谱图", fontsize=15)
# plt.specgram(wav_data, Fs=24000, scale_by_freq=True, sides='default', cmap="jet")
# plt.xlabel('秒/s', fontsize=15)
# plt.ylabel('频率/Hz', fontsize=15)
#
# plt.subplot(2, 2, 2)
# plt.title("波形图", fontsize=15)
# time = np.arange(0, len(wav_data)) * (1.0 / fs)
# plt.plot(time, wav_data)
# plt.xlabel('秒/s', fontsize=15)
# plt.ylabel('振幅', fontsize=15)
#
# plt.tight_layout()
# plt.show()
#
# # ########### 画图
# plt.subplot(2, 2, 1)
# plt.title("语谱图", fontsize=15)
# plt.specgram(wav_data1, Fs=24000, scale_by_freq=True, sides='default', cmap="jet")
# plt.xlabel('秒/s', fontsize=15)
# plt.ylabel('频率/Hz', fontsize=15)
#
# plt.subplot(2, 2, 2)
# plt.title("波形图", fontsize=15)
# time = np.arange(0, len(wav_data1)) * (1.0 / fs)
# plt.plot(time, wav_data1)
# plt.xlabel('秒/s', fontsize=15)
# plt.ylabel('振幅', fontsize=15)
#
# plt.tight_layout()
# plt.show()

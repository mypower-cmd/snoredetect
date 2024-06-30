import numpy as np
import pandas as pd
import wave
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa

# pd.set_option('display.max_rows', None)#显示全部行
# pd.set_option('display.max_columns', None)#显示全部列


# 将.wav放在pycharm工程目录下,rb为只读
f = wave.open(r"new_snore_0000.wav", "rb")
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
# 返回声道数、量化位数、采样频率、采样点数
print(params[:4])
# readframes返回的是字符串类型的数据
str_data = f.readframes(nframes)
# 将字符串转换为int类型的数组，我的sampwidth为2，所以转换为int16
signal = np.frombuffer(str_data, dtype=np.int16)
# 归一化
signal = signal / (max(abs(signal)))

# ------------------预加重---------------------

# 预加重，kp=0.85
signal_add = np.append(signal[0], signal[1:] - 0.85 * signal[:-1])

# 画时间轴
# time = np.arange(0, nframes) / (1.0 * framerate)

# plt.figure(figsize=(20, 10))
# plt.subplot(2, 1, 1)
#
# plt.plot(time, signal)
# plt.xlabel('time')
# plt.ylabel("Amplitude")
# plt.grid()
# plt.subplot(2, 1, 2)
#
# plt.plot(time, signal_add)
# plt.xlabel('time')
# plt.ylabel("Amplitude")
# plt.grid()
# plt.show()

# ----分帧------

wlen = 2048  # 窗长/帧长
inc = 2048  # 帧移
N = 2048  # 每帧信号的数据长度
# 计算帧数nf
nf = int(np.ceil((1.0 * nframes - wlen + inc) / inc))
# 所有帧加起来总的铺平后的长度
pad_len = int((nf - 1) * inc + wlen)
# 不够的点用0填补，类似于FFT中的扩充数组操作
zeros = np.zeros(pad_len - nframes)
# 填补后的信号记为pad_signal
pad_signal = np.concatenate((signal, zeros))
# 相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵
indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),(wlen, 1)).T
# 将indices转化为矩阵
indices = np.array(indices, dtype=np.int32)
# 最终得到帧数*帧长形式的信号数据
frames = pad_signal[indices]
# 画图
a = frames[0:1]
time = np.arange(0, wlen) * (1.0 / framerate)
plt.figure(figsize=(10, 4))
plt.plot(time, a[0])
plt.grid()
plt.show()

# -------------------加窗--------------------

# 汉宁窗函数
plt.figure(figsize=(6, 2))
# numpy
plt.plot(np.hanning(512))
# scipy.signal
# plt.plot(signal.hanning(512))
plt.grid()
plt.show()
# 先调用窗函数
win = np.hanning(wlen)
print(frames,len(frames),len(frames[0]))
x = frames[0:1]  # 我选取其中某一帧的数据提取PLP系数，即为frames的某一行
y = win * x[0]  # 得到加窗后的信号
# print(x)
# print(x[0])
# print(y)
time = np.arange(0, wlen) * (1.0 / framerate)
plt.figure(figsize=(10, 4))
plt.plot(time, y)
plt.grid()
plt.show()

# -----先定义了两个公式，bark_change为线性频率坐标转换为Bark坐标，equal_loudness为等响度曲线--------

def bark_change(x):
    return 6 * np.log10(x / (1200 * np.pi) + ((x / (1200 * np.pi)) ** 2 + 1) ** 0.5)


def equal_loudness(x):
    return ((x ** 2 + 56.8e6) * x ** 4) / ((x ** 2 + 6.3e6) ** 2 * (x ** 2 + 3.8e8))


# 做fft，默认为信号的长度
a = np.fft.fft(y)
# 求fft变换结果的模的平方，只取a的一半
b = np.square(abs(a[0:N]))
# 频率分辨率
df = framerate / N
# 只取大于0部分的频率
i = np.arange(N)
# 得到实际频率坐标
freq_hz = i * df
# 得到该帧信号的功率谱
plt.plot(freq_hz, b)
plt.show()

# 转换为角频率
freq_w = 2 * np.pi * np.array(freq_hz)
# 再转换为bark频率
freq_bark = bark_change(freq_w)
# 选取的临界频带数量一般要大于10，覆盖常用频率范围，这里我选取了15个中心频点
point_hz = [250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400]
# 转换为角频率
point_w = 2 * np.pi * np.array(point_hz)
# 转换为bark频率
point_bark = bark_change(point_w)
# 构造15行512列的矩阵，每一行为一个滤波器向量
bank = np.zeros((15, N))
# 构造15维频带能量向量
filter_data = np.zeros(15)

# -------构造滤波器组---------

for j in range(15):
    for k in range(N):
        omg = freq_bark[k] - point_bark[j]
        if -1.3 < omg < -0.5:
            bank[j, k] = 10 ** (2.5 * (omg + 0.5))
        elif -0.5 < omg < 0.5:
            bank[j, k] = 1
        elif 0.5 < omg < 2.5:
            bank[j, k] = 10 ** (-1.0 * (omg - 0.5))
        else:
            bank[j, k] = 0
    filter_data[j] = np.sum(b * bank[j])  # 滤波后将该段信号相加，最终得到15维的频带能量
    plt.plot(freq_hz, bank[j])
    plt.xlim(0, 20000)
    plt.xlabel('hz')
plt.show()  # 画滤波器组

equal_data = equal_loudness(point_w) * filter_data
# 强度-响度转换
cubic_data = equal_data ** 0.33

# ---做30点的ifft，得到30维PLP向量------------
plp_data = np.fft.ifft(cubic_data, 30)
# print(plp_data)
# print(abs(plp_data))

# ---用librosa直接求传统LPC系数，需要0.7以上版本---
# plp = librosa.lpc(abs(plp_data), order=15)  # 要求输入参数为浮点型，经过ifft得到的plp_data有复数，因此要取abs
# h1 = 1.0 / np.fft.fft(plp, 1024)
# spec_envelope_plp = 10 * np.log10(abs(h1[0:512]))
# lpc = librosa.lpc(y, order=15)
# h2 = 1.0 / np.fft.fft(lpc, 1024)
# spec_envelope_lpc = 10 * np.log10(abs(h2[0:512]))
# plt.plot(spec_envelope_plp, 'b', spec_envelope_lpc, 'r')
# plt.show()

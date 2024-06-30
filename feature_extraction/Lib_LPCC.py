import os
import numpy
import pandas as pd
import scipy.io.wavfile as wav
from spafe.features import lpc
from spafe.utils.preprocessing import SlidingWindow

train_data_dir = "D:/Project/snore/Aug_Snore/train/"
val_data_dir = "D:/Project/snore/Aug_Snore/val/"
test_data_dir = "D:/Project/snore/Aug_Snore/test/"

train_save_dir = "D:/Project/snore/Snore_dataset_LPCC/train/"
val_save_dir = "D:/Project/snore/Snore_dataset_LPCC/val/"
test_save_dir = "D:/Project/snore/Snore_dataset_LPCC/test/"


def transform(data_base_dir, data_save_dir):
    # err = []
    for cur_dir in os.listdir(data_base_dir):
        for wav_file_name in os.listdir(data_base_dir + cur_dir):
            pwd_wav_file = data_base_dir + cur_dir + "/" + wav_file_name
            # 读取音频文件
            rate, sig = wav.read(pwd_wav_file)
            # 提取LPCC特征
            # try:
            lpccs = lpc.lpcc(sig,
                             fs=24000,
                             order=13,
                             pre_emph=True,
                             pre_emph_coeff=0.85,
                             window=SlidingWindow(0.0853, 0.0853, "hanning"))
            # except numpy.linalg.LinAlgError:
            #     err.append(wav_file_name.split('.')[0])
            #     continue
            # else:
            # 输出LPCC特征
            # print(lpccs)
            # print(len(lpccs), len(lpccs[0]))
            # 取2-13维数据
            lpccs = lpccs[:, 1:]
            # LPCC的数据不足补0
            lpccs = numpy.array(lpccs)
            lpccs = lpccs.T
            new_rol = numpy.zeros((12, 1), dtype=float)
            while len(lpccs[0]) < 42:
                lpccs = numpy.append(lpccs, new_rol, axis=1)
            lpccs = lpccs.T
            # 保存成csv文件
            df = pd.DataFrame(lpccs)
            save_dir = data_save_dir + cur_dir + "/" + wav_file_name.split('.')[0]
            df.to_csv(save_dir + '.csv')
            print(wav_file_name.split('.')[0])
# print("Transition finished")
# print("--------------err list-------------")
# for i in err:
#     print(i)


if __name__ == '__main__':
    # transform(test_data_dir, test_save_dir)
    # transform(train_data_dir, train_save_dir)
    transform(val_data_dir, val_save_dir)

import os
import numpy
import pandas as pd
import scipy.io.wavfile as wav
from spafe.features import rplp
from spafe.utils.preprocessing import SlidingWindow

train_data_dir = "D:/Project/snore/Aug_Snore/train/"
val_data_dir = "D:/Project/snore/Aug_Snore/val/"
test_data_dir = "D:/Project/snore/Aug_Snore/test/"

train_save_dir = "D:/Project/snore/Snore_dataset_PLP/train/"
val_save_dir = "D:/Project/snore/Snore_dataset_PLP/val/"
test_save_dir = "D:/Project/snore/Snore_dataset_PLP/test/"


def transform(data_base_dir, data_save_dir):
    err = []
    # curr_train_dirs = os.listdir(data_base_dir)
    # for curr_train_dir in curr_train_dirs:
    # 鼾声和非鼾声数据集
    for curr_train_dir in range(3):
        pwd_train_dir = data_base_dir + str(curr_train_dir)
        # 鼾声数据名
        wav_file_names = os.listdir(pwd_train_dir)
        for wav_file_name in wav_file_names:
            pwd_wav_file = pwd_train_dir + "/" + wav_file_name
            # 读取音频文件
            rate, sig = wav.read(pwd_wav_file)
            # 提取PLP特征
            # try:
            plps = rplp.plp(sig,
                            fs=24000,
                            order=14,
                            pre_emph=True,
                            pre_emph_coeff=0.85,
                            window=SlidingWindow(0.0853, 0.0853, "hanning"),
                            nfft=2048,
                            fbanks=None)
            # except numpy.linalg.LinAlgError:
            #     err.append(wav_file_name.split('.')[0])
            #     continue
            # else:
            # 输出PLP特征
            # print(plps)
            # print(len(plps), len(plps[0]))
            # 取2-13维数据
            plps = plps[:, 2:]
            # PLP的数据不足补0
            plps = numpy.array(plps)
            plps = plps.T
            new_rol = numpy.zeros((12, 1), dtype=float)
            while len(plps[0]) < 42:
                plps = numpy.append(plps, new_rol, axis=1)
            plps = plps.T
            # 保存成csv文件
            df = pd.DataFrame(plps)
            save_dir = data_save_dir + str(curr_train_dir) + "/" + wav_file_name.split('.')[0]
            df.to_csv(save_dir + '.csv')
            print(wav_file_name.split('.')[0])

# print("Transition finished")
# print("--------------err list-------------")
# for i in err:
#     print(i)


if __name__ == '__main__':
    transform(test_data_dir, test_save_dir)
    transform(train_data_dir, train_save_dir)
    transform(val_data_dir, val_save_dir)

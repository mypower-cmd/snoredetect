import os
import subprocess

input_train_path = r"D:\Project\snore\Aug_Snore_3_5\train"
input_val_path = r"D:\Project\snore\Aug_Snore_3_5\val"
output_train_path = r"D:\Project\snore\Aug_Snore\train"
output_val_path = r"D:\Project\snore\Aug_Snore\val"

input_path = r"D:\Project\snore\temp1"
output_path = r"D:\Project\snore\temp"

def resample(input_path, output_path):
    # for i in range(3):
    #     new_input_path = input_path + '\\' + str(3)
    #     new_output_path = output_path + '\\' + str(4)
    folder_names = os.listdir(input_path)
    for folder in folder_names:
        for i, file in enumerate(os.listdir(input_path + "\\" + folder)):
            file1 = input_path + "\\" + folder +  '\\' + file
            file2 = output_path + "\\" + folder +  '\\' + folder + "_%04d.wav" % i
            cmd = "ffmpeg -i " + file1 + " -ar 24000 -ac 1 " + file2  # ffmpeg -i 输入文件 -ar 采样率  输出文件
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    resample(input_path, output_path)
    # resample(input_val_path, output_val_path)

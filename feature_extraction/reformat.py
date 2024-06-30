import os
import subprocess

# input_train_path = r"D:\Project\snore\Aug_Snore_3_5\train"
# input_val_path = r"D:\Project\snore\Aug_Snore_3_5\val"
# output_train_path = r"D:\Project\snore\Aug_Snore\train"
# output_val_path = r"D:\Project\snore\Aug_Snore\val"

input_path = r"D:\Project\snore\all_dataset\temp1"


def reformat(input_path, output_path):
    # for i in range(3):
    #     new_input_path = input_path + '\\' + str(i)
        # new_output_path = input_path + '\\' + str(i)
    for file in os.listdir(input_path):
        file1 = input_path + '\\' + file
        file2 = output_path + '\\' + file.split(".")[0] + ".wav"
        cmd = "ffmpeg -i " + file1 + " -ar 24000 -ac 1 " + file2  # ffmpeg -i 输入文件 -ar 采样率  输出文件
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    reformat(input_path, input_path)

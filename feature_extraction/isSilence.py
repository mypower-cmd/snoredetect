import pandas as pd
import os


# ffmpeg -i hello.mp3 -af silencedetect=noise=0.0001 -f null -
import os
import subprocess

input_path = r"D:\Project\snore\Snore_dataset\train\2"
remove_path = r"D:\Project\snore\Aug_Snore\train\2"

def isSilence(input_path, remove_path):
    for file in os.listdir(input_path):
        cnt = 0
        file1 = input_path +  '\\' + file
        file2 = remove_path +  '\\' + file.split(".")[0] + ".wav"
        data = pd.read_csv(file1, header=None, index_col=0).values[1:]
        for i in range(42):
            if data[i][0] == 0:
                cnt += 1
        if cnt >= 10:
            print(f"{file.split('.')[0]} is removed.")
            os.remove(file1)
            os.remove(file2)

if __name__ == '__main__':
    isSilence(input_path, remove_path)
    # resample(input_val_path, output_val_path)

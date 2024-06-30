# ffmpeg -i input.wav -ss 00:00:05 -t 00:00:10 output.wav
import os
import subprocess

input_path = r"D:\Project\snore\temp\speech"
output_path = r"D:\Project\snore\temp1\speech"

def reformat(input_path, output_path):
    for i in range(3):
        for file in os.listdir(input_path):
            file1 = input_path + '\\' + file
            file2 = output_path + '\\' + ".".join(file.split(".")[:-1]) + "_%s" % (i+1) + ".wav"
            cmd = "ffmpeg -i " + file1 + " -c:a copy -ss 00:00:%0d -t 00:00:03 " % (i*3) + file2
            subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    reformat(input_path, output_path)

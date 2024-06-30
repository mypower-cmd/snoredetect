import os

file_dir = r"D:\Project\snore\Aug_Snore\val\2"


def rename(path):
    file_name = os.listdir(path)
    for i, name in enumerate(file_name):
        print(file_name[i], "is end.")
        os.rename(path + "\\" + name, path + "\\" + "quite_%05d.wav" % i)


if __name__ == '__main__':
    rename(file_dir)

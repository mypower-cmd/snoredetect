import os
import shutil
import random

old_dir0 = r"D:\Project\snore\Aug_Snore\train\0"
old_dir1 = r"D:\Project\snore\Aug_Snore\train\1"
old_dir2 = r"D:\Project\snore\Aug_Snore\train\2"

test_dir0 = r"D:\Project\snore\Aug_Snore\test\0"
val_dir0 = r"D:\Project\snore\Aug_Snore\val\0"

test_dir1 = r"D:\Project\snore\Aug_Snore\test\1"
val_dir1 = r"D:\Project\snore\Aug_Snore\val\1"

test_dir2 = r"D:\Project\snore\Aug_Snore\test\2"
val_dir2 = r"D:\Project\snore\Aug_Snore\val\2"


def reset(old_dir, new_dir, size):
    old_file = os.listdir(old_dir)
    new_file = random.sample(old_file, len(old_file) // size)
    for file in new_file:
        src = os.path.join(old_dir, file)
        dst = os.path.join(new_dir, file)
        shutil.move(src, dst)
    print("切分完成")


if __name__ == '__main__':
    # 非鼾声
    reset(old_dir0, val_dir0, 5)
    reset(old_dir0, test_dir0, 8)
    # 鼾声
    reset(old_dir1, val_dir1, 5)
    reset(old_dir1, test_dir1, 8)
    # 安静
    reset(old_dir2, val_dir2, 5)
    reset(old_dir2, test_dir2, 8)

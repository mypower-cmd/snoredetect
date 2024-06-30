import time
from process_data import get_data_labels
import pandas as pd
import numpy as np
from config import cfg
from tensorflow import keras
import tensorflow as tf
import sys

sys.path.append("..")

label_test_txt = "./data_list/label_test.txt"

list_result = "./result.txt"

model = keras.models.load_model(cfg.save_mode_path)


def prediction_list(list_table):
    cnt = 0
    count = 0
    i = 0
    total1 = 0
    filenames, true_labels = get_data_labels(list_table)
    total = len(filenames)
    with open(list_result, "w") as f:
        for filename, true_label in zip(filenames, true_labels):
            i = i + 1
            csv_data = pd.read_csv(filename,
                                   header=None, index_col=0).values[1:]
            # csv_data = 10e5 * np.float32(
            #     csv_data)  # 别直接用float, 因为numpy和python互不相关

            # Tensorflow NHWC
            input_wav = tf.expand_dims(csv_data, axis=0)
            data = tf.expand_dims(input_wav, axis=3)

            predictions = model.predict(data)
            score = tf.nn.softmax(predictions[0])
            predic_cls = np.argmax(score)
            predic_scr = np.max(score)

            print("\r进度：{:6.2f}%".format((float(i / total * 100))), end=' ')
            # print("%-20s" % filename.split("/")[-1].split(".")[0], end=' ')
            # time.sleep(0.01)
            # print(int(true_label), predic_cls, "%.2f" % predic_scr, predic_cls == int(true_label))

            if int(true_label) == 1:
                total1 += 1
                if predic_cls == int(true_label):
                    count = count + 1
            if predic_cls == int(true_label):
                cnt = cnt + 1
            else:
                print("%-20s" % filename.split("/")[-1].split(".")[0], end=' ')
                time.sleep(0.01)
                print(int(true_label), predic_cls, "%.2f" % predic_scr, predic_cls == int(true_label))

            write_info = "%-60s" % filename + " " + true_label + "\t" + str(
                predic_cls) + "\t" + str(predic_scr) + "\n"
            f.write(write_info)
    print("鼾声准确率：%.2f%%" % (100 * count / total1))
    print("总体准确率：%.2f%%" % (100 * cnt / total))

if __name__ == '__main__':
    prediction_list(label_test_txt)

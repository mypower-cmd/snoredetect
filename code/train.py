import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import numpy as np
import datetime
from tensorflow.keras.callbacks import (ReduceLROnPlateau, TensorBoard)
import sys
from process_data import gen_data_batch
from model import *
import os
import pandas as pd
sys.path.append("..")
# model = cnn_model()
# model = cnn_rnn_model()
# model = rnn_model()
import matplotlib
matplotlib.use('TkAgg')


class MySetLR(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        # 学习率控制
        learning_rate = model.optimizer.lr.numpy().astype(np.float64)
        if epoch < 8:
            lr = learning_rate
        else:
            lr = learning_rate * np.exp(0.1 * (8 - epoch))
        K.set_value(model.optimizer.lr, lr)
        print(
            'Epoch %03d: LearningRateScheduler reducing learning rate to %s.' %
            (epoch + 1, lr))


#配置GPU，限制GPU内存增长
def GPU_Config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def train_and_val():
    # 配置GPU
    GPU_Config()

    # 创建模型
    model = rnn_model()


    list_num_samples = np.load(cfg.train_num_samples_path)  # 读取
    cfg.num_samples = list_num_samples
    print("num_samples:", cfg.num_samples)

    dataset = gen_data_batch(cfg.train_dataset_path,
                                   cfg.batch_size,
                                   1,
                                   is_training=True)

    train_num =int((cfg.num_samples*(1 - cfg.val_rate - cfg.test_rate))//cfg.batch_size)
    val_num = int((cfg.num_samples * cfg.val_rate)//cfg.batch_size)
    test_num = int((cfg.num_samples * cfg.test_rate)//cfg.batch_size)
    print("train_num", train_num)
    print("val_num", val_num)
    print("test_num", test_num)
    train_data = dataset.take(train_num)
    temp_data = dataset.skip(train_num)
    val_data = temp_data.take(val_num)
    test_data = temp_data.skip(val_num)

    optimizer = tf.keras.optimizers.Adam(lr=cfg.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # callbacks = [
    #     ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1),
    #     MySetLR(),
    #     TensorBoard(log_dir=cfg.log_dir, histogram_freq=1),
    #

    #创建保存文件路径
    folder_path = "../model/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_path = "../model/" + "sleep_classify_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".h5"

    # 配置回调函数
    callbacks = [
        # ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ]

    history = model.fit(train_data,
                        epochs=cfg.epochs,
                        batch_size=cfg.batch_size,
                        callbacks=callbacks,
                        validation_data=val_data,
                        shuffle=True,
                        verbose=1,
                        validation_freq=1,
                        # steps_per_epoch=train_num,
                        # validation_steps=val_num
                        )
    #创建保存文件路径
    folder_path = "../results/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pd.DataFrame(history.history).to_csv('../results/training_log.csv', index=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 画出训练和测试的准确率和损失值
    epochs_range = range(cfg.epochs)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # 保存图片12
    result_png = "../results/" + "result_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()



    print('-----------test--------')
    test_loss, test_acc = model.evaluate(test_data, steps=test_num, verbose=1)
    print('测试准确率：', test_acc)
    print('测试损失', test_loss)

    # 模型保存
    final_model_path = "../model/" + "sleep_classify_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + "final.h5"
    model.save(final_model_path)


if __name__ == '__main__':
    train_and_val()

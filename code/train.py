import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import numpy as np
import datetime
from tensorflow.keras.callbacks import (ReduceLROnPlateau, TensorBoard)
import sys
from process_data import gen_data_batch
from model import *

sys.path.append("..")
model = cnn_model()
# model = cnn_rnn_model()
# model = rnn_model()


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


def train_and_val():
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
    train_dataset = gen_data_batch(cfg.train.dataset,
                                   cfg.batch_size,
                                   cfg.epochs,
                                   is_training=True)
    val_dataset = gen_data_batch(cfg.val.dataset,
                                 cfg.batch_size,
                                 is_training=False)
    optimizer = tf.keras.optimizers.Adam(lr=cfg.train.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1),
        MySetLR(),
        TensorBoard(log_dir=cfg.log_dir, histogram_freq=1),
    ]

    history = model.fit(train_dataset,
                        epochs=cfg.epochs,
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        steps_per_epoch=cfg.train.num_samples // cfg.batch_size,
                        validation_steps=cfg.val.num_samples // cfg.batch_size)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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

    result_png = "./result_img/" + "result_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()

    test_loss, test_acc = model.evaluate(val_dataset,
                                         steps=(cfg.val.num_samples //
                                                cfg.batch_size),
                                         verbose=1)
    print("test_acc =", test_acc)
    print("test_loss =", test_loss)

    model.save(cfg.save_mode_path)


if __name__ == '__main__':
    train_and_val()

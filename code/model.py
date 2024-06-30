from tensorflow import keras
from config import cfg
from tensorflow.keras import layers, regularizers

# 基础的CNN模型
def cnn_model():
    input_data = keras.Input(shape=(42, 12, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu')(input_data)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    model_output = layers.Dense(cfg.num_classes, activation='softmax')(x)
    model = keras.Model(input_data, model_output)
    return model


# 三种正则化：L1/L2/dropout/批量归一化层（要注意，这个主要是提升速度，解决数据深层传递问题，提高泛化能力，同时附带正则化）
def cnn_model_l1_l2_BN():
    input_data = keras.Input(shape=(42, 12, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
    # x = layers.BatchNormalization()(x)  # 加入批量归一化层
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    # x = layers.BatchNormalization()(x)  # 加入批量归一化层

    # L1L2两种正则化方式
    # x = layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l1(0.01))(x)
    # x = layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)

    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    # x = layers.BatchNormalization()(x)  # 加入批量归一化层

    x = layers.Dropout(0.5)(x)  # 弃权技术正则化
    model_output = layers.Dense(cfg.num_classes, activation='softmax')(x)  # 有些说这里要做少分类的话，可以换个优化函数（softmax），有些人又说跟relu冲突
    model = keras.Model(input_data, model_output)
    return model


# rnn模型
# def rnn_model():
#     input_wav = keras.Input(shape=(42, 12))
#     x = layers.LSTM(32, return_sequences=True)(input_wav)
#     x = layers.LSTM(64, return_sequences=True)(x)
#     x = layers.LSTM(64, return_sequences=True)(x)
#     x = layers.LSTM(64, return_sequences=True)(x)
#     model_output = layers.LSTM(cfg.num_classes,
#                                activation='relu',
#                                return_sequences=False)(x)
#     model = keras.Model(input_wav, model_output)
#     return model


# 轻量化的rnn
def rnn_model():
    input_wav = keras.Input(shape=(42, 12))
    x = layers.LSTM(16, return_sequences=True)(input_wav)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    model_output = layers.Dense(cfg.num_classes, activation='softmax')(x)
    model = keras.Model(input_wav, model_output)
    return model

# cnn_rnn混合模型
# 新增优化，减少卷积核
def cnn_rnn_model():
    input_wav = keras.Input(shape=(42, 12, 1))
    x = layers.Conv2D(16, (3, 3), activation='relu')(input_wav)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.Reshape(target_shape=(18 * 3, 32))(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    model_output = layers.Dense(cfg.num_classes)(x)
    model = keras.Model(input_wav, model_output)
    return model

# 优化混合模型，这里主要是想要回溯之前的数据，然后看看做正则化能不能改变之前的过拟合现象
def cnn_rnn_model_BN():
    input_wav = keras.Input(shape=(42, 12, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_wav)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape(target_shape=(18 * 3, 64))(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    model_output = layers.Dense(cfg.num_classes)(x)
    model = keras.Model(input_wav, model_output)
    return model


mymodel = cnn_model()
mymodel.summary()

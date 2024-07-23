from process_data import *
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

path_tfrecords = cfg.train_dataset_path
predict_model = tf.keras.models.load_model('../model/sleep_classify_0723_212738.h5', compile=True) #  注意这儿得compile需要设置为true，如果你不设置你需要多一步compile的过程。
test_dataset = gen_valdata_batch(path_tfrecords, cfg.batch_size)

# dataset = test_dataset
# list_num_samples = np.load(cfg.train_num_samples_path)  # 读取
# cfg.num_samples = list_num_samples
# print("num_samples:", cfg.num_samples)
# train_num = int((cfg.num_samples * (1 - cfg.val_rate - cfg.test_rate)) // cfg.batch_size)
# val_num = int((cfg.num_samples * cfg.val_rate) // cfg.batch_size)
# test_num = int((cfg.num_samples * cfg.test_rate) // cfg.batch_size)
# print("train_num", train_num)
# print("val_num", val_num)
# print("test_num", test_num)
# train_data = dataset.take(train_num)
# temp_data = dataset.skip(train_num)
# val_data = temp_data.take(val_num)
# test_data = temp_data.skip(val_num)
# test_dataset = test_data

print('-----------test--------')
test_loss, test_acc = predict_model.evaluate(test_dataset, batch_size=cfg.batch_size, verbose=1)
print('测试准确率：', test_acc)
print('测试损失', test_loss)

y_predict = predict_model.predict(test_dataset, verbose=1)
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_predict = y_predict.flatten().tolist()
# print(y_predict)
y_true = get_truelabel(path_tfrecords)
# print(y_true)

# truelabel = []
# for data, label in test_dataset:
#     label = np.int64(label)
#     truelabel.extend(label)
# y_true = truelabel

confusionmatrix = confusion_matrix( y_true, y_predict ) #行的标签代表真实值，列的标签代表预测值
print(confusionmatrix)
TP0 = confusionmatrix[0][0]
TN0 = confusionmatrix[1][1] + confusionmatrix[1][2] + confusionmatrix[2][1] + confusionmatrix[2][2]
FP0 = confusionmatrix[1][0] + confusionmatrix[2][0]
FN0 = confusionmatrix[0][1] + confusionmatrix[0][2]

TP1 = confusionmatrix[1][1]
TN1 = confusionmatrix[0][0] + confusionmatrix[0][2] + confusionmatrix[2][0] + confusionmatrix[2][2]
FP1 = confusionmatrix[0][1] + confusionmatrix[2][1]
FN1 = confusionmatrix[1][0] + confusionmatrix[1][2]

TP2 = confusionmatrix[2][2]
TN2 = confusionmatrix[0][0] + confusionmatrix[0][1] + confusionmatrix[1][0] + confusionmatrix[1][1]
FP2 = confusionmatrix[0][2] + confusionmatrix[1][2]
FN2 = confusionmatrix[2][0] + confusionmatrix[2][1]

Accuracy = (TP0 + TP1 + TP2) / sum(sum(confusionmatrix))
print("Accuracy:", Accuracy)

P0 = TP0 / (TP0 + FP0)
P1 = TP1 / (TP1 + FP1)
P2 = TP2 / (TP2 + FP2)
Precision = (P0 + P1 + P2) / 3
print("Precision:", Precision)

R0 = TP0 / (TP0 + FN0)
R1 = TP1 / (TP1 + FN1)
R2 = TP2 / (TP2 + FN2)
Recall = (R0 + R1 + R2) / 3
print("Recall:", Recall)

F1_Score = 2 * Precision * Recall / (Precision + Recall)
print("F1_Score:", F1_Score)

confusionmatrix_percent = []
for matrix in confusionmatrix:
    matrix_sum = np.sum(confusionmatrix, axis=0)
    # print(matri?x)
    matrix = matrix / matrix_sum
    # print(matrix)
    confusionmatrix_percent.append(matrix)
# print(confusionmatrix_percent)

# x_ticks = ["supine", "right latericumbent", "left latericumbent", "prone"]
# y_ticks = ["supine", "right latericumbent", "left latericumbent", "prone"]  # 自定义横纵轴

# x_ticks = ["supine", "right", "left", "prone"]
# y_ticks = ["supine", "right", "left", "prone"]  # 自定义横纵轴

x_ticks = ["env", "snore", "quite"]
y_ticks = ["env", "snore", "quite"]  # 自定义横纵轴

ax = sns.heatmap(confusionmatrix_percent, xticklabels=x_ticks, yticklabels=y_ticks, annot=True, cbar=None, cmap='Blues', fmt='.3f')
# ax = sns.heatmap(confusionmatrix, xticklabels=x_ticks, yticklabels=y_ticks, annot=True, cbar=None, cmap='Blues', fmt='g')
ax.set_title('Confusion Matrix')  # 图标题
ax.set_xlabel('True Label')  # x轴标题
ax.set_ylabel('Predict Label')
plt.show()
figure = ax.get_figure()
figure.savefig('../results/sns_heatmap.jpg')  # 保存图片
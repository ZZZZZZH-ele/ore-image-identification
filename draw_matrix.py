#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 16:53
# @Author  : wangbm
# @FileName: draw_matrix.py
# @Info:

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#weights_path = r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords\resnet50_val\\'
weights_path = r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords\resnet50att_val\\'

font = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 18,
        }

classes = ['host', 'ore']
all_preds = np.load(weights_path + 'matrix_pred.npy')
all_label = np.load(weights_path + 'matrix_label.npy')
plt.figure()
report = classification_report(all_label, all_preds, target_names=classes, output_dict=True)
print(report)
print('val accuracy:', report['accuracy'])
val_accuracy = report['accuracy']
confusion = confusion_matrix(all_label, all_preds)
normalized_confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
normalized_confusion = np.round(normalized_confusion, 3)
# 绘制热度图
plt.imshow(normalized_confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes,fontsize=13,fontweight='bold')
plt.yticks(indices, classes,fontsize=13,fontweight='bold')
colorbar = plt.colorbar()
ticklabels = colorbar.ax.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(13)  # 设置字体大小为12
    label.set_weight('bold')  # 设置字体加粗
plt.xlabel('Predicted label',font)
plt.ylabel('True label',font)
# 显示数据
for first_index in range(len(normalized_confusion)):
    for second_index in range(len(normalized_confusion[first_index])):
        if first_index==second_index:
            color='white'
        else:
            color='black'
        plt.text(second_index, first_index, normalized_confusion[first_index][second_index], color=color,verticalalignment='center', horizontalalignment='center',fontsize=13,fontweight='bold')
    #plt.title('Confusion Matrix of' + ' ' + 'Mean Teacher ResNet50 improved Accuracy: ' + str(round(report['accuracy'], 5)))
    #plt.title('Confusion Matrix of' + ' ' + 'Mean Teacher ResNet50 base Accuracy: ' + str(round(report['accuracy'], 5)))
    #plt.title('ResNet50 Accuracy: ' + str(round(report['accuracy'], 5)),font)
    plt.title('ECA-ResNet50 Accuracy: ' + str(round(report['accuracy'], 5)),font)
plt.tight_layout()
# 显示图片
plt.savefig(weights_path + 'matrix.eps', dpi=350, format='eps')
plt.savefig(weights_path + 'matrix.png', dpi=350, format='png')
plt.savefig(weights_path + 'matrix.svg', dpi=350, format='svg')
plt.show()




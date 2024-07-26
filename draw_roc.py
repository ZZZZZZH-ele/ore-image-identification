#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

weights_path = r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords\resnet50_val\\'

classes = ['host', 'ore']
FPR = np.load(weights_path + 'FPR.npy')
TPR = np.load(weights_path + 'TPR.npy')
AUC = auc(FPR, TPR)
print('AUC:', AUC)

# ROC曲线绘制
plt.figure()
# plt.title('Mean Teacher ROC CURVE ResNet50 improved AUC={:.5f}'.format(AUC))
plt.title('ROC CURVE ResNet50 AUC={:.5f}'.format(AUC))
# plt.title('ROC CURVE ECA-ResNet50 AUC={:.5f}'.format(AUC))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.plot(FPR, TPR, color='g', linestyle=':',
         label='ResNet50 AUC: ' + str(round(AUC, 5)))
         # label='ECA-ResNet50 AUC: ' + str(round(AUC, 5)))
         # label='Mean Teacher ResNet50 improved AUC: ' + str(round(AUC, 5)))
# 增加标签颜色说明
plt.legend()
plt.plot([0, 1], [0, 1], color='m', linestyle='--')
# 显示图片
plt.savefig(weights_path + 'roc.eps', dpi=350, format='eps')
plt.savefig(weights_path + 'roc.png', dpi=350, format='png')
plt.savefig(weights_path + 'roc.svg', dpi=350, format='svg')
plt.show()

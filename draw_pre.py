import matplotlib.pyplot as plt
import numpy as np

confusion = np.array([[218,10],[12,270]])
normalized_confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
normalized_confusion = np.round(normalized_confusion, 3)
font = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 18,
        }
classes = ['host', 'ore']
indices = range(len(normalized_confusion))
plt.figure()
plt.imshow(normalized_confusion, cmap='Blues')
plt.xticks(indices, classes,fontsize=13,fontweight='bold')
plt.yticks(indices, classes,fontsize=13,fontweight='bold')
colorbar = plt.colorbar()
ticklabels = colorbar.ax.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(13)  # 设置字体大小为12
    label.set_weight('bold')  # 设置字体加粗
plt.xlabel('Predicted label',font)
plt.ylabel('True label',font)
for i in range(len(normalized_confusion)):
    for j in range(len(normalized_confusion[i])):
        if i == j:
            color = 'white'
        else:
            color = 'black'
        plt.text(j, i, normalized_confusion[i][j], ha='center', va='center', color=color, fontsize=15, fontweight='bold')

plt.title('Testing result',font)
plt.savefig(r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords\testing result\\'+ 'matrix.png', dpi=350, format='png')
plt.show()
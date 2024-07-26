import cv2
import numpy as np

# 绘制热图色条棒
color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), cv2.COLORMAP_JET)
color_map = cv2.flip(color_map, 0)  # 翻转色条棒，使其自下往上渐变
color_map = cv2.resize(color_map, (30, 512), interpolation=cv2.INTER_NEAREST)  # 调整色条棒的尺寸
color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)  # 将色条棒颜色通道顺序转为RGB

# 保存带刻度色条棒的图片
cv2.imwrite('colorbar_with_scale.png', color_map)
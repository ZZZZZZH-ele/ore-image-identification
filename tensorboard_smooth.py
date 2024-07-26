#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def smooth(csv_path, weight=0.8):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int, 'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    save.to_csv(csv_path.replace('.csv', '_smooth.csv'))


if __name__ == '__main__':
    smooth(r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords\resnet50_r50_adam_batch4_loss.csv')
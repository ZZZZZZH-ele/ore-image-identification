#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/7/30 18:24
# @Author  : wangbm
# @FileName: cam_show.py
# @Info:
import torch
import numpy as np
import cv2
from PIL import Image

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

def show(args, feature, i, save_path, val_images_path, org_img):
    feature = torch.mean(feature, dim=1)
    feature = feature[0, :, :]
    img = feature.cpu().detach().numpy()
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    heat_map = cv2.resize(img, (480, 640))
    # 下面读取原图进行叠加显示
    # 首先将原图转移到cpu以及numpy格式
    org_img = org_img.detach().cpu().numpy()
    if args.lights.count('_') == 0:
        processe_org_img = cv_imread(val_images_path[i])
        processe_cam_img = cv2.addWeighted(processe_org_img, 1, heat_map, 0.5, 0)
        show_img = np.hstack([processe_org_img, processe_cam_img])
    elif args.lights.count('_') == 1:
        processe_org_img_1 = cv_imread(val_images_path[i])
        processe_org_img_2 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[1]))
        processe_cam_img_1 = cv2.addWeighted(processe_org_img_1, 1, heat_map, 0.5, 0)
        processe_cam_img_2 = cv2.addWeighted(processe_org_img_2, 1, heat_map, 0.5, 0)
        show_img = np.hstack([processe_cam_img_1, processe_cam_img_2])
    elif args.lights.count('_') == 2:
        processe_org_img_1 = cv_imread(val_images_path[i])
        processe_org_img_2 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[1]))
        processe_org_img_3 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[2]))
        processe_cam_img_1 = cv2.addWeighted(processe_org_img_1, 1, heat_map, 0.5, 0)
        processe_cam_img_2 = cv2.addWeighted(processe_org_img_2, 1, heat_map, 0.5, 0)
        processe_cam_img_3 = cv2.addWeighted(processe_org_img_3, 1, heat_map, 0.5, 0)
        show_img = np.hstack([processe_cam_img_1, processe_cam_img_2, processe_cam_img_3])
    elif args.lights.count('_') == 3:
        processe_org_img_1 = cv_imread(val_images_path[i])
        processe_org_img_2 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[1]))
        processe_org_img_3 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[2]))
        processe_org_img_4 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[3]))
        processe_cam_img_1 = cv2.addWeighted(processe_org_img_1, 1, heat_map, 0.5, 0)
        processe_cam_img_2 = cv2.addWeighted(processe_org_img_2, 1, heat_map, 0.5, 0)
        processe_cam_img_3 = cv2.addWeighted(processe_org_img_3, 1, heat_map, 0.5, 0)
        processe_cam_img_4 = cv2.addWeighted(processe_org_img_4, 1, heat_map, 0.5, 0)
        show_img = np.hstack([processe_cam_img_1, processe_cam_img_2, processe_cam_img_3, processe_cam_img_4])
    elif args.lights.count('_') == 4:
        processe_org_img_1 = cv_imread(val_images_path[i])
        processe_org_img_2 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[1]))
        processe_org_img_3 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[2]))
        processe_org_img_4 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[3]))
        processe_org_img_5 = cv_imread(val_images_path[i].replace(args.lights.split('_')[0], args.lights.split('_')[4]))
        processe_cam_img_1 = cv2.addWeighted(processe_org_img_1, 1, heat_map, 0.5, 0)
        processe_cam_img_2 = cv2.addWeighted(processe_org_img_2, 1, heat_map, 0.5, 0)
        processe_cam_img_3 = cv2.addWeighted(processe_org_img_3, 1, heat_map, 0.5, 0)
        processe_cam_img_4 = cv2.addWeighted(processe_org_img_4, 1, heat_map, 0.5, 0)
        processe_cam_img_5 = cv2.addWeighted(processe_org_img_5, 1, heat_map, 0.5, 0)
        show_img = np.hstack([processe_cam_img_1, processe_cam_img_2, processe_cam_img_3, processe_cam_img_4, processe_cam_img_5])

    cv2.imencode('.png', show_img)[1].tofile(save_path + '\\' + val_images_path[i].split('\\')[-1].split('.')[0] + '.png')




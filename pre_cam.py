#!/usr/bin/python3
# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import logging
import argparse
import os
import torch
from models.model_resnet_cam import resnet50
from models.model_resnet_att_cam import eca_resnet50
from PIL import Image
import torchvision.transforms as transforms
logger = logging.getLogger(__name__)
import numpy as np
import cv2

def setup(args):
    # Prepare model
    if args.dataset == "mineral":
        num_classes = 2
    if args.method == 'resnet50':
        model = resnet50(num_classes=num_classes, include_top=True)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    elif args.method == 'resnet50att':
        print('we use resnet50att')
        model = eca_resnet50(num_classes=num_classes)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    else:
        print('model chose error！')
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def valid(args, model):
    # Validation!
    # Prepare dataset
    image_names = glob.glob(args.data_root + '\\*.jpg')
    model.eval()

    with torch.no_grad():
        i = 0
        for name in image_names:
            base_name = name.split('\\')[-1]
            processe_org_img = cv2.imread(name)
            x = Image.open(name)
            transf = transforms.ToTensor()
            x = transf(x)
            x = x.to(args.device)
            x = x.unsqueeze(0)
            logits, cam_feature = model(x)
            i += 1
            preds = torch.argmax(logits, dim=-1)
            print(str(name) + ' image mineral type: ', preds.item())
            # 针对cam的处理
            feature = torch.mean(cam_feature, dim=1)
            feature = feature[0, :, :]
            img = feature.cpu().detach().numpy()
            pmin = np.min(img)
            pmax = np.max(img)
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
            img = img.astype(np.uint8)  # 转成unit8
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
            heat_map = cv2.resize(img, (512, 512))
            processe_cam_img = cv2.addWeighted(processe_org_img, 1, heat_map, 0.5, 0)
            show_img = np.hstack([processe_org_img, processe_cam_img])
            os.makedirs(args.cam_save, exist_ok=True)
            cv2.imwrite(os.path.join(args.cam_save, base_name), show_img)


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=['mineral'],
                        default="mineral",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default=r'C:\zh\jupyter-notebook\qh')
    parser.add_argument('--cam_save', type=str, default=r'C:\zh\jupyter-notebook\qh2')
    parser.add_argument('--method', type=str, default="resnet50att",
                        choices=["resnet50att", "resnet50"])
    parser.add_argument("--test_pretrain_weights", type=str, default=r"C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords\r"
                                                                     r"esnet50att_r50esa_adam_batch4\_checkpoint_4200.bin",
                        help="test_pretrain_weights path")
    parser.add_argument("--img_size", default=[512, 512], type=int,
                        help="Resolution size")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.nprocs = torch.cuda.device_count()
    # Model & Tokenizer Setup
    args, model = setup(args)
    valid(args, model)


#!/usr/bin/python3
# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import argparse
import os
import random
import shutil
import numpy as np
from datetime import timedelta
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from models.modeling import VisionTransformer, CONFIGS
from models.model_resnet_for_cam import resnet50, resnet101
from models.model_mobilenet import MobileNetV2
from models.model_densenet import densenet121
from utils.data_utils import get_loader, get_loader_cam, get_loader_cam_order
from utils.cam_show import show
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "melasma":
        num_classes = 2

    if args.method == 'transfg':
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes,
                                  smoothing_value=args.smoothing_value)
        # if args.pretrained_model is not None:
        #     pretrained_model = torch.load(args.pretrained_model)['model']
        #     model.load_state_dict(pretrained_model)
    elif args.method == 'resnet50':
        model = resnet50(num_classes=num_classes, include_top=True)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    elif args.method == 'mobilenet':
        print('we use mobilenet')
        model = MobileNetV2(num_classes=num_classes)
        model_weight_path = "/slurm_data/wangbm/Project/Classifications/Classification/Test6_mobilenet/mobilenet_v2.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'classifier' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.method == 'densenet':
        model = densenet121(num_classes=num_classes)
        model_weight_path = "/slurm_data/wangbm/Project/Classifications/Classification/Test8_densenet/densenet121.pth"
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'classifier' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print('模型选择异常！')
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def valid(args, model):
    # Validation!
    # Prepare dataset
    train_loader, test_loader, train_images_path, val_images_path = get_loader_cam_order(args)

    target_layers = [ model.layer4[-1] ]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    model.eval()
    all_preds, all_label = [], []
    num_test = len(test_loader)
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=False,
                          disable=args.local_rank not in [-1, 0],
                          ncols=80,
                          position=0, leave=True)
    '''
    下面针对二分类验证集进行loss曲线绘制，混淆矩阵fig生成，AUC计算与绘制
    '''
    acc = 0.0
    TP_num, FP_num, TN_num, FN_num = 0, 0, 0, 0
    pred_list = []
    label_list = []

    with torch.no_grad():
        i = 0
        for x, y, for_cam_images, org_name in epoch_iterator:
            x = x.to(args.device)
            y = y.to(args.device)
            logits, feature = model(x)
            show(feature, i, '_cam', save_path=r'C:\zh\ore_image_recoginition\OtherProjects\chongqing\cam_images', val_images_path=val_images_path,
                 for_cam_images=for_cam_images, org_name=org_name)
            i += 1
            preds = torch.argmax(logits, dim=-1)
            '''
            下面是针对二分类问题进行各指标计算，表格中敏感性等计算
            '''
            acc += torch.eq(preds, y).sum().item()
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )
            # 针对二分类问题
            outputs = F.softmax(logits, dim=1)
            pred_ = outputs[:, 1].tolist()
            _, preds = torch.max(outputs, 1)
            labels = y.cpu().numpy()
            preds = preds.cpu().numpy()
            pred_list.append(pred_)
            label_list.append(labels.tolist())
            TP_num += (preds[np.where(labels.any() == 1)] > 0.5).sum()
            FP_num += (preds[np.where(labels == 0)] > 0.5).sum()
            TN_num += (preds[np.where(labels == 0)] < 0.5).sum()
            FN_num += (preds[np.where(labels.any() == 1)] < 0.5).sum()

    # 下面这个是二分类，也就是正常类和异常类
    print('acc:', acc / num_test)
    print('----------二分类结果----------------')
    FPR, TPR, threshold = roc_curve(label_list, pred_list, pos_label=1)

    AUC = auc(FPR, TPR)
    print('AUC:', AUC)
    Sensitive = TP_num / (TP_num + FN_num)
    Specificity = TN_num / (FP_num + TN_num)
    Precision = TP_num / (TP_num + FP_num)
    ACC = (TP_num + TN_num) * 100 / (TP_num + TN_num + FP_num + FN_num)
    print('Sensitive:', Sensitive)
    print('Specificity:', Specificity)
    print('Precision:', Precision)
    print('ACC:', ACC)

    all_preds, all_label = all_preds[0], all_label[0]
    # ROC曲线绘制
    plt.figure()
    plt.title('ROC CURVE')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.plot(FPR, TPR, color='darkorange', label=args.step + ' AUC: ' + str(round(AUC, 3)))
    plt.legend()
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.show()
    classes_for_confusion_matrix = ['without', 'with']
    plt.savefig(args.this_val_save_path + args.step + 'best_auc.png', dpi=350, format='png')
    plt.savefig(args.this_val_save_path + args.step + 'best_auc.eps', dpi=350, format='eps')
    # 将上一步的保存下来
    np.save(args.this_val_save_path + args.step + 'best_FPR.npy', FPR)
    np.save(args.this_val_save_path + args.step + 'best_TPR.npy', TPR)
    # 下面进行混淆矩阵的绘制
    plt.figure()
    np.save(args.this_val_save_path + args.step + 'best_matrix_label.npy', all_label)
    np.save(args.this_val_save_path + args.step + 'best_matrix_pred.npy', all_preds)
    report = classification_report(all_label, all_preds, target_names=classes_for_confusion_matrix)
    print(report)
    confusion = confusion_matrix(all_label, all_preds)
    print(confusion)
    # 绘制热度图
    plt.imshow(confusion, cmap=plt.cm.Oranges)
    indices = range(len(confusion))
    plt.xticks(indices, classes_for_confusion_matrix)
    plt.yticks(indices, classes_for_confusion_matrix)
    plt.colorbar()
    plt.xlabel('True')
    plt.ylabel('Pred')
    # 显示数据
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(second_index, first_index, confusion[first_index][second_index])
    # 显示图片
    plt.savefig(args.this_val_save_path + args.step + 'best_matrix.eps', dpi=350, format='eps')
    plt.savefig(args.this_val_save_path + args.step + 'best_matrix.png', dpi=350, format='png')
    plt.show()


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "BC", "nabirds", "INat2017", 'melasma'],
                        default="melasma",
                        help="Which dataset.")
    parser.add_argument("--lights", default="normal",
                        choices=['normal', 'normal_red', 'normal_red_brown',
                                 'normal_red_brown_blue', 'normal_red_brown_blue_black'],
                        help="Which dataset.")
    #改了
    parser.add_argument('--data_root', type=str, default=r'C:\zh\ore_image_recoginition\mineralcls\processeddata\test')
    parser.add_argument('--data_path', type=str, default=r'C:\zh\ore_image_recoginition\mineralcls\processeddata\test')
    parser.add_argument('--save_root_path', type=str,
                        default='/slurm_data/wangbm/Project/Classifications/Classification/TransFG')
    parser.add_argument('--method', type=str, default="resnet50",
                        choices=["transfg", "resnet50", "densenet", "mobilenet"])
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--test_pretrain_weights", type=str, default=r"C:\zh\ore_image_recoginition\OtherProjects\chongqing\weights\normal_save_"
                                                                     r"100_weights_checkpoint_3300.bin",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--step", type=str, default="transfg_red_brown_blue_black",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--this_val_save_path", type=str, default="/slurm_data/wangbm/Project/Class"
                                                                  "ifications/Classification/TransFG/preds_paper_normal_cam/",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_dir", type=str, default=r"/slurm_data/wangbm/Project/Classifications/Classificati"
                                                              r"on/TransFG/weights/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=r"/slurm_data/wangbm/Project/Classifications/Classificati"
                                                              r"on/TransFG/weights/imagenet21k_ViT-B_16.npz",
                        help="load pretrained model")
    parser.add_argument("--output_dir",
                        default=r"C:\zh\ore_image_recoginition\OtherProjects\chongqing\weights\output_weights", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=[480, 640], type=int,
                        help="Resolution size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="num_workers")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=500, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=1000000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()

    # 创建模型的文件夹

    args.this_val_save_path = args.this_val_save_path + '\\' + args.step + '\\'
    os.makedirs(args.this_val_save_path, exist_ok=True)
    # 移动模型权重
    shutil.copy(args.test_pretrain_weights, args.this_val_save_path + args.step + '.bin')


    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    valid(args, model)

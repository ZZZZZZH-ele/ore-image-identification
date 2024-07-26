# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
import argparse
import os
import random
import numpy as np
import time
from datetime import timedelta
import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.model_resnet import resnet50, resnet34, resnet101, resnet152
from models.model_resnet_att import eca_resnet50
# from models.model_vgg import vgg
# from models.model_alexnet import AlexNet
# from models.model_efficientnet import efficientnet_b0
# from models.model_googlenet import GoogLeNet
# from models.model_mobilenet import MobileNetV2
# from models.model_densenet import densenet121
# from models.model_shufflenet import shufflenet_v2_x0_5
# from models.model_swin import swin_tiny_patch4_window7_224, swin_base_patch4_window7_224
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_mt import get_loader_mt
from utils import ramps
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
from itertools import chain


logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def update_ema_variables(model, model_teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    return model, model_teacher

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(this_val_save_path, "%s_checkpoint.bin" % args.name)
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)

def save_model_step(args, model, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(this_val_save_path, "_checkpoint_" + str(global_step) + '.bin')
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", this_val_save_path)

def setup(args):
    # Prepare model
    if args.dataset == "mineral":
        num_classes = 2
    else:
        print('数据集异常')

    if args.method == 'resnet50':
        print('we use resnet50')
        model = resnet50(num_classes=num_classes, include_top=True)
    elif args.method == 'resnet50att':
        print('we use resnet50att')
        model = eca_resnet50(num_classes=num_classes)
    else:
        print('模型选择异常！')

    ema_model = deepcopy(model)
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    model.to(args.device)
    ema_model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model, ema_model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(epoch, 200)

def valid(args, model, writer, test_loader, global_step, best_acc):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    # 下面几个针对二分类的问题
    TP_num, FP_num, TN_num, FN_num = 0, 0, 0, 0
    pred_list = []
    label_list = []
    num_test = len(test_loader)
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=False,
                          disable=args.local_rank not in [-1, 0],
                          ncols=80,
                          position=0, leave=True, mininterval=10)
    loss_fct = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        i = 0
        for x, y in epoch_iterator:

            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())
            i += 1
            preds = torch.argmax(logits, dim=-1)
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

            TP_num += (preds[np.where(labels.any() == 1 or labels.any() == 2 or labels.any() == 3)] > 0.5).sum()
            FP_num += (preds[np.where(labels == 0)] > 0.5).sum()
            TN_num += (preds[np.where(labels == 0)] < 0.5).sum()
            FN_num += (preds[np.where(labels.any() == 1 or labels.any() == 2 or labels.any() == 3)] < 0.5).sum()

            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # 下面这个是二分类，也就是正常类和异常类
    print('----------二分类结果----------------')
    FPR, TPR, threshold = roc_curve(list(chain.from_iterable(label_list)), list(chain.from_iterable(pred_list)), pos_label=1)
    # fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
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
    classes = ['host', 'ore']

    plt.figure()
    report = classification_report(all_label, all_preds, target_names=classes, output_dict=True)
    print(report)
    print('val accuracy:', report['accuracy'])
    val_accuracy = report['accuracy']
    writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)
    # 选择最好的结果进行保存
    if val_accuracy > best_acc:
        np.save(this_val_save_path + str(global_step) + '_matrix_label.npy', all_label)
        np.save(this_val_save_path + str(global_step) + '_matrix_pred.npy', all_preds)
        confusion = confusion_matrix(all_label, all_preds)
        # 绘制热度图
        plt.imshow(confusion, cmap=plt.cm.Blues)
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('Pred')
        plt.ylabel('True')
        # 显示数据
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(second_index, first_index, confusion[first_index][second_index])
        # 显示图片
        plt.savefig(this_val_save_path + str(global_step) + '_matrix.eps', dpi=350, format='eps')
        plt.savefig(this_val_save_path + str(global_step) + '_matrix.png', dpi=350, format='png')
        # plt.show()
    else:
        pass
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)

    return val_accuracy

def consistency_loss(logits_w1, logits_w2):
    logits_w2 = logits_w2.detach()
    assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(torch.softmax(logits_w1, dim=-1), torch.softmax(logits_w2, dim=-1), reduction='mean')

def train(args, model, ema_model):
    """ Train the model """
    summary_save_path = args.save_root_path + '/' + args.method + '_' + args.name
    os.makedirs(summary_save_path, exist_ok=True)
    writer = SummaryWriter(summary_save_path)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    # Prepare dataset
    train_loader_labeled, train_loader_unlabeled, test_loader = get_loader_mt(args)

    label_iter = iter(train_loader_labeled)
    #
    loss_function = nn.CrossEntropyLoss()
    # Prepare optimizer and scheduler
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=0.9,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate)

    t_total = args.num_steps
    # if args.decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    ema_model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()

    global_step, best_acc = 0, 0
    start_time = time.time()
    cur_itrs = 0

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    while True:
        model.train()
        ema_model.train()

        all_preds, all_label = [], []

        epoch_iterator = tqdm(train_loader_unlabeled,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=False,
                              disable=args.local_rank not in [-1, 0],
                              ncols=80,
                              mininterval=10)

        for step, ((img1_ul, img2_ul), target_no_label) in enumerate(epoch_iterator):
            cur_itrs += 1
            try:
                (img1, img2), target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(train_loader_labeled)
                (img1, img2), target_label, = next(label_iter)

            batch_size_labeled = img1.shape[0]
            input1 = Variable(torch.cat([img1, img1_ul]).to(args.device))
            input2 = Variable(torch.cat([img2, img2_ul]).to(args.device))
            target = Variable(target_label.to(args.device))

            output = model(input1)

            with torch.no_grad():
                ema_output = ema_model(input2)

            unsup_loss = consistency_loss(output, ema_output)
            out_x = output[:batch_size_labeled]
            sup_loss = ce_loss(out_x, target.long())
            consistency_weight = get_current_consistency_weight(cur_itrs//150)

            loss = sup_loss + consistency_weight * unsup_loss  # 损失为 有标签的交叉熵损失+ 一致性损失(基于平滑性假设，一个模型对于 一个输入及其变形应该保持一致性）
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item() * args.gradient_accumulation_steps)
            optimizer.step()
            model, ema_model = update_ema_variables(model, ema_model, args.ema_m, global_step)  # 更新模型


            y = target
            preds = torch.argmax(output[:batch_size_labeled], dim=-1)

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

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)

                if global_step > 0 and global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid(args, model, writer, test_loader, global_step, best_acc)
                    if args.local_rank in [-1, 0]:
                        if best_acc < accuracy:
                            save_model_step(args, model, global_step)
                            best_acc = accuracy
                        logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        train_accuracy = simple_accuracy(all_preds, all_label)
        # train_accuracy = accuracy.cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default='r50esa_adam_batch4',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=['mineral'],   # 数据集名字
                        default="mineral",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default=r'C:\zh\ore_image_recoginition\mineralcls\processeddata')   # 文件夹路径
    parser.add_argument('--data_path', type=str, default=r'C:\zh\ore_image_recoginition\mineralcls\processeddata')
    parser.add_argument('--save_root_path', type=str,
                        default=r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords')    # 保存路径
    parser.add_argument('--val_save_root_path', type=str,
                        default=r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords')
    parser.add_argument('--method', type=str, default="resnet50",
                        choices=["resnet50att", "resnet50"])
    parser.add_argument("--img_size", default=[512, 512], type=int,
                        help="Resolution size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="num_workers")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=200, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for adaM.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=6000, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--unsup_warm_up", default=0.4, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--ema_m", default=0.999, type=float,
                        help="ema_m.")

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
    args, model, ema_model = setup(args)
    # 定义保存验证结果的路径
    val_save_root_path = args.val_save_root_path
    this_val_save_path = val_save_root_path + '/' + args.method + '_' + args.name + '/'
    os.makedirs(this_val_save_path, exist_ok=True)
    # Training
    train(args, model, ema_model)

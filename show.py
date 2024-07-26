# -*- coding: utf-8 -*-
import os
import pandas as pd
from matplotlib import pyplot as plt


def loss_visualize(epoch_loss_res, value_loss_res, epoch_loss_att, value_loss_att,
                   epoch_acc_res, value_acc_res, epoch_acc_att, value_acc_att):
    # plt.style.use('seaborn-bright')
    print(plt.style.available)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("LOSS&ACC_EPOCH_CURVE")
    plt.plot(epoch_loss_att, value_loss_att, label='ResNet50 improved Training Loss', color='g',
             linestyle='-')
    plt.plot(epoch_acc_att, value_acc_att, label='ResNet50 improved Validation Accuracy', color='g', linestyle='-.')

    plt.plot(epoch_loss_res, value_loss_res, label='ResNet50 based Training Loss', color='b', linestyle='-')
    plt.plot(epoch_acc_res, value_acc_res, label='ResNet50 based Validation Accuracy', color='b', linestyle='-.')
    plt.legend()
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS&ACC')
    plt.grid()
    plt.savefig(os.path.join(res_dir, r'class_acc_loss_epoch_.eps'), dpi=350, format='eps')
    plt.savefig(os.path.join(res_dir, r'class_acc_loss_epoch_.png'), dpi=350, format='png')
    plt.savefig(os.path.join(res_dir, r'class_acc_loss_epoch_.svg'), dpi=350, format='svg')

    plt.show()


def read_value(train_df):
    epoch = train_df['Step']
    value = train_df['Value']
    return epoch, value


if __name__ == "__main__":
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 15,
            }
    root_dir = os.getcwd()
    file_dir = os.path.join(root_dir, r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords')
    res_dir = os.path.join(root_dir, r'C:\zh\ore_image_recoginition\mineralcls\mineralcodes\trainingrecords')
    loss_resnet50 = pd.read_csv(os.path.join(file_dir, 'resnet50_r50_adam_batch4_loss_smooth.csv'))
    loss_resnet50att = pd.read_csv(os.path.join(file_dir, 'resnet50att_r50esa_adam_batch4_loss_smooth.csv'))
    acc_resnet50 = pd.read_csv(os.path.join(file_dir, 'resnet50_r50_adam_batch4_acc_smooth.csv'))
    acc_resnet50att = pd.read_csv(os.path.join(file_dir, 'resnet50att_r50esa_adam_batch4_acc_smooth.csv'))
    epoch_loss_res, value_loss_res = read_value(loss_resnet50)
    epoch_loss_att, value_loss_att = read_value(loss_resnet50att)
    epoch_acc_res, value_acc_res = read_value(acc_resnet50)
    epoch_acc_att, value_acc_att = read_value(acc_resnet50att)

    # loss_visualize(epoch_loss_res, value_loss_res, epoch_loss_att, value_loss_att,
    #                epoch_acc_res, value_acc_res, epoch_acc_att, value_acc_att)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Validation Accuracy",font)
    # plt.plot(epoch_loss_att, value_loss_att, label='ResNet50 improved Training Loss', color='g',
    #          linestyle='-')
    plt.plot(epoch_acc_att, value_acc_att, label='ECA-ResNet50 ', color='r', linestyle='-')

    # plt.plot(epoch_loss_res, value_loss_res, label='ResNet50 based Training Loss', color='b', linestyle='-')
    plt.plot(epoch_acc_res, value_acc_res, label='ResNet50', color='b', linestyle='-')
    plt.legend()
    plt.xlabel('Iterations',font)
    plt.ylabel('Accuracy',font)
    plt.grid()
    plt.savefig(os.path.join(res_dir, r'class_acc_loss_epoch_.eps'), dpi=350, format='eps')
    plt.savefig(os.path.join(res_dir, r'class_acc_loss_epoch_.png'), dpi=350, format='png')
    plt.savefig(os.path.join(res_dir, r'class_acc_loss_epoch_.svg'), dpi=350, format='svg')

    plt.show()
    # loss_visualize(epoch_loss_res, value_loss_res, epoch_loss_att, value_loss_att,
    #                epoch_acc_res, value_acc_res, epoch_acc_att, value_acc_att)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Traning Loss",font)
    # plt.plot(epoch_loss_att, value_loss_att, label='ResNet50 improved Training Loss', color='g',
    #          linestyle='-')
    plt.plot(epoch_loss_att, value_loss_att, label='ECA-ResNet50 ', color='r', linestyle='-')

    # plt.plot(epoch_loss_res, value_loss_res, label='ResNet50 based Training Loss', color='b', linestyle='-')
    plt.plot(epoch_loss_res, value_loss_res, label='ResNet50', color='b', linestyle='-')
    plt.legend()
    plt.xlabel('Iterations',font)
    plt.ylabel('Loss',font)
    plt.grid()
    plt.savefig(os.path.join(res_dir, r'class_loss_epoch_.eps'), dpi=350, format='eps')
    plt.savefig(os.path.join(res_dir, r'class_loss_epoch_.png'), dpi=350, format='png')
    plt.savefig(os.path.join(res_dir, r'class_loss_epoch_.svg'), dpi=350, format='svg')

    plt.show()
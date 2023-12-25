import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
import pandas as pd

def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
    fig = plt.figure()		# 创建图形实例
    ax = plt.subplot(111)		# 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                        fontdict={'weight': 'bold', 'size': 7})

    plt.xticks()		# 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig
if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt_file = '../data/slc_val_3.txt'
    batch_size = 30
    cate_num = 8

    spe_transform = transforms.Compose([
        # transform_data.Normalize_spe_xy(),
        transform_data.Numpy2Tensor()
    ])

    img_transform = transforms.Compose([
        transform_data.Normalize_img(),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = slc_dataset.SLC_img_spe4D_single(txt_file=txt_file,
                                        img_dir='../data/slc_data/',
                                        spe_dir='../data/spexy_data_32/',
                                        img_transform=img_transform,
                                        spe_transform=spe_transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    net_joint = network.SLC_joint2_single(cate_num)
    net_joint.load_state_dict(torch.load('../model_single/slc_joint_deeper_single3_F_epoch2831.pth'))
    net_joint.to(device)
    # net_joint = network.ResNet18_ssl(cate_num)
    # net_joint.load_state_dict(torch.load('./实验结果_Res/all_res/slc_joint_deeper_img__FR_epoch551.pth'))
    # net_joint.to(device)


    acc_num = 0.0
    data_num = 0
    val_loss = 0.0

    label2name = pd.read_csv('../data/catename2label_cate8.txt')

    net_joint.eval()
    iter_val = iter(dataloader)
    y_score = np.zeros([0,8])
    y_label = np.zeros([0], dtype=int)
    for j in range(len(dataloader)):
        val_data = next(iter_val)
        val_img = val_data['img'].to(device)
        val_spe = val_data['spe'].to(device)
        val_label = val_data['label'].to(device)
        val_output = net_joint(val_spe, val_img)
        # _, val_output = net_joint(val_spe, val_img, val_label)
        # val_loss = loss_func(val_output, val_label)
        y_label = np.concatenate((y_label, val_label.cpu().data.numpy()), axis=0)
        y_score = np.concatenate((y_score, val_output.cpu().data.numpy()))
        _, pred = torch.Tensor.max(val_output, 1)
        acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
        data_num += val_label.size()[0]
        #val_loss += loss_func(val_output, val_label).item()

    val_acc = acc_num / data_num
    print(y_score.shape)
    print(val_acc)
    

    result = TSNE(learning_rate=50, n_components=2).fit_transform(y_score)
    fig = plot_embedding(result, y_label, 't-SNE Embedding of digits')
    plt.savefig("../model/3_single.png")
    plt.show()

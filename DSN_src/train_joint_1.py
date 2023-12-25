import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
from torch import nn
from learning_schedule import param_setting_jointmodel2
from torch import optim
import argparse


def get_pretrained(img_model, net_joint):
    # spe_mapping = {'encoder1':'pre_spe.0',
    #                'encoder2':'pre_spe.2',
    #                'encoder3':'pre_spe.3',
    #                'encoder4':'pre_spe.5'}
    img_mapping = {'conv1':'pre_img.0',
                   'bn1':'pre_img.1',
                   'layer1':'pre_img.3',
                   'layer2':'pre_img.4'}
    # for key in spe_model.keys():
    #     if key[:8] in spe_mapping.keys():
    #         net_joint.state_dict()[spe_mapping[key[:8]] + key[8:]].data.copy_(spe_model[key])
    for key in img_model.keys():
        if key[:5] in img_mapping.keys():
            #print(key)
            net_joint.state_dict()[img_mapping[key[:5]] + key[5:]].data.copy_(img_model[key])
        elif key[:3] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:3]] + key[3:]].data.copy_(img_model[key])
            #print(key)
        elif key[:6] in img_mapping.keys():
            #print(key)
            net_joint.state_dict()[img_mapping[key[:6]] + key[6:]].data.copy_(img_model[key])
    return net_joint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train')

    parser.add_argument('--training_dataset', default=1)
    args = parser.parse_args()
    datasetnum = args.training_dataset


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt_file = {'train': '../data/slc_train_' + str(datasetnum) + '.txt',
                'val': '../data/slc_val_' + str(datasetnum) + '.txt'}
    #batch_size = {'train': 256,
    #              'val': 64}
    batch_size = {'train': 256,
                  'val': 64}
    cate_num = 8
    save_model_path = '../model_multi1/slc_joint_deeper_' + str(datasetnum) + '_F_'

    spe_transform = transforms.Compose([
        # transform_data.Normalize_spe_xy(),
        transform_data.Numpy2Tensor()
    ])

    img_transform = transforms.Compose([
        transform_data.Normalize_img(),
        transform_data.Numpy2Tensor_img(3)
    ])


    dataset = {x : slc_dataset.SLC_img_spe4D(txt_file=txt_file[x],
                                            img_dir='../data/slc_data/',
                                            spe_dir_32='../data/spexy_data_32/',
                                            spe_dir_48='../data/spexy_data_48/',
                                            spe_dir_16='../data/spexy_data_16/',
                                            img_transform=img_transform,
                                            spe_transform=spe_transform)
               for x in ['train', 'val']}

    train_count_dict = {}
    for i in range(cate_num):
        train_count_dict[i] = len(dataset['train'].data.loc[dataset['train'].data['label'] == i])

    loss_weight = [(1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
                       for i in range(cate_num)]

    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset['train'],
                                      batch_size=batch_size['train'],
                                      sampler=ImbalancedDatasetSampler(dataset['train']),
                                      num_workers=0)
    dataloaders['val'] = DataLoader(dataset['val'],
                                    batch_size=batch_size['val'],
                                    shuffle=True,
                                    num_workers=0)

    img_model = torch.load('../model/resnet18_I_nwpu_cate45_tsx_level1_cate7_col36_imb_ce+topk+.pth')
    #img_model = torch.load('../model/resnet18_tsx_mstar_epoch7.pth')
    # spe_model = torch.load('../model/slc_spexy_cae_2.pth')
    net_joint = network.SLC_joint2(cate_num)
    net_joint = get_pretrained(img_model, net_joint)
    #net_joint.load_state_dict(torch.load('../model_multi7/slc_joint_deeper_7_F_epoch7000.pth'))

    net_joint.to(device)


    epoch_num=7000
    i = 0
    parameter_list = param_setting_jointmodel2(model=net_joint)

    optimizer = optim.SGD(parameter_list, weight_decay=0.0005)
    lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
    print(lr_list)
    loss_weight = torch.Tensor(loss_weight).to(device)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight)

    f_acc = open('../model/accuracy1.txt','w')
    f_loss = open('../model/loss1.txt','w')
    for epoch in range(epoch_num):

        for data in dataloaders['train']:
            net_joint.train()
            optimizer.zero_grad()

            img_data = data['img'].to(device)
            spe_data_32 = data['spe_32'].to(device)
            spe_data_48 = data['spe_48'].to(device)
            spe_data_16 = data['spe_16'].to(device)
            labels = data['label'].to(device)
            output = net_joint(spe_data_32, spe_data_48,spe_data_16, img_data)
            #output, _ = net_joint(spe_data, img_data,labels)
            # reg = 0.5 * 0.0005 * sum(p.data.norm() ** 2 for p in net_joint.parameters())
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            i += 1

        acc_num = 0.0
        data_num = 0
        val_loss = 0.0

        print('epoch ' + str(epoch + 1) + '\titer ' + str(i) + '\tloss ', loss.item())
        f_loss.write(str(loss.item()))
        net_joint.eval()
        iter_val = iter(dataloaders['val'])
        for j in range(len(dataloaders['val'])):
            val_data = next(iter_val)
            val_img = val_data['img'].to(device)
            val_spe_32 = val_data['spe_32'].to(device)
            val_spe_48 = val_data['spe_48'].to(device)
            val_spe_16 = val_data['spe_16'].to(device)
            val_label = val_data['label'].to(device)
            val_output = net_joint(val_spe_32,val_spe_48,val_spe_16, val_img)
            #_,val_output = net_joint(val_spe, val_img,val_label)
            prob, pred = torch.Tensor.max(val_output, 1)
            acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
            data_num += val_label.size()[0]
            val_loss += loss_func(val_output, val_label).item()

        val_loss /= len(dataloaders['val'])
        val_acc = acc_num / data_num
        print(val_acc)
        f_acc.write(str(val_acc.item()) + '\n')
        if epoch % 100 == 0:

            torch.save(net_joint.state_dict(), save_model_path + 'epoch' + str(epoch + 1) + '.pth')

    f_acc.close()
    f_loss.close()


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

    for key in img_model.keys():
        if key[:5] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:5]] + key[5:]].data.copy_(img_model[key])
        elif key[:3] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:3]] + key[3:]].data.copy_(img_model[key])
        elif key[:6] in img_mapping.keys():
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
    batch_size = {'train': 64,
                  'val': 16}
    cate_num = 8
    save_model_path = '../model/slc_joint_deeper_img_' + str(datasetnum) + '_FR_'


    img_transform = transforms.Compose([
        transform_data.Normalize_img(),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = {x : slc_dataset.SLC_img(txt_file=txt_file[x],
                                       root_dir='../data/slc_data/',
                                       transform=img_transform,
                                    )
               for x in ['train', 'val']}

    train_count_dict = {}
    for i in range(cate_num):
        train_count_dict[i] = len(dataset['train'].data.loc[dataset['train'].data['label'] == i])
        #print(train_count_dict){0: 330, 1: 176, 2: 285, 3: 183, 4: 312, 5: 332, 6: 302, 7: 310}

    loss_weight = [(1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
                       for i in range(cate_num)]
    #print(loss_weight)#[0.9737347853939783, 1.0526585522101217, 0.9967969250480461, 1.0490711082639332, 0.9829596412556053, 0.972709801409353, 0.9880845611787316, 0.9839846252402307]
    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset['train'],
                                      batch_size=batch_size['train'],
                                      sampler=ImbalancedDatasetSampler(dataset['train']),
                                      num_workers=0)
    print(dataloaders['train'])
    dataloaders['val'] = DataLoader(dataset['val'],
                                    batch_size=batch_size['val'],
                                    shuffle=True,
                                    num_workers=0)

    # img_model = torch.load('../model/tsx.pth')
    # spe_model = torch.load('../model/slc_spexy_cae_2.pth')
    net_joint = network.SLC_joint2_img(cate_num)
    # net_joint = get_pretrained(img_model, net_joint)
    # net_joint.load_state_dict(torch.load('../model/slc_joint_deeper_img_3_F_epoch300.pth'))

    net_joint.to(device)


    epoch_num = 500
    i = 0
    parameter_list = param_setting_jointmodel2(model=net_joint)

    optimizer = optim.SGD(parameter_list, lr=0.01, weight_decay=0.0005)
    lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
    loss_weight = torch.Tensor(loss_weight).to(device)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight)

    for epoch in range(epoch_num):

        for data in dataloaders['train']:
            net_joint.train()
            optimizer.zero_grad()

            img_data = data['data'].to(device)
            labels = data['label'].to(device)
            output = net_joint(img_data)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            i += 1


        acc_num = 0.0
        data_num = 0
        val_loss = 0.0

        print('epoch ' + str(epoch + 1) + '\titer ' + str(i) + '\tloss ', loss.item())
        net_joint.eval()
        iter_val = iter(dataloaders['val'])
        for j in range(len(dataloaders['val'])):
            val_data = next(iter_val)
            val_img = val_data['data'].to(device)
            val_label = val_data['label'].to(device)
            val_output = net_joint(val_img)

            # val_loss = loss_func(val_output, val_label)
            _, pred = torch.Tensor.max(val_output, 1)
            acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
            data_num += val_label.size()[0]
            val_loss += loss_func(val_output, val_label).item()

        val_loss /= len(dataloaders['val'])
        val_acc = acc_num / data_num
        print(val_acc)


        torch.save(net_joint.state_dict(), save_model_path + 'epoch' + str(epoch + 1) + '.pth')



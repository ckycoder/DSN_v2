import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import transform_data
import network
from slc_dataset import SLC_spe_xy
from torch import optim, nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

txt_file = {'train': '../data/slc_cae_train.txt',
           'val': '../data/slc_cae_val.txt'}
batch_size = {'train': 512,
              'val': 100}
cate_num = 5
save_model_path = '../model_16/slc_spexy_cae_3_'


data_transforms = transforms.Compose([
    transform_data.Normalize_spe_xy(),
    transform_data.Numpy2Tensor_img(1)
])


dataset = {x : SLC_spe_xy(txt_file=txt_file[x], spe_dir='../data/spe_data_16/',sublook=16,
                                       spe_transform = data_transforms)
           for x in ['train', 'val']}
print(len(dataset['train']))
dataloader = {x : DataLoader(dataset[x],
                              batch_size=batch_size[x],
                              shuffle=True,
                              num_workers=0)
               for x in ['train', 'val']}

net = network.SLC_spexy_CAE_16()
optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0005)
loss_func = nn.MSELoss()
# loss_func = losses.SARLoss()

epoch_num = 100
i = 0
net.to(device)

params = list(net.parameters())

# fig = plt.subplots(2, 5)
# fig.show()

iter_val = iter(dataloader['val'])
for epoch in range(epoch_num):
    for sample in dataloader['train']:
        net.train()
        data = sample['spe']
        optimizer.zero_grad()
        output = net(data.to(device))
        loss = loss_func(output, data.to(device))
        loss.backward()
        optimizer.step()
        # print(params[0].grad)

        if i % 10 == 0:
            print('epoch ' + str(epoch+1) + '\titer ' + str(i) + '\tloss ', loss.item())
            net.eval()
            try:
                val_sample = next(iter_val)
            except StopIteration:
                iter_val = iter(dataloader['val'])
                val_sample = next(iter_val)

            val_data = val_sample['spe']

            val_output = net(val_data.to(device))
            val_loss = loss_func(val_output, val_data.to(device))

            if i % 100 == 0:

                val_imgs = torch.cat((val_data, val_output.cpu()))
                val_imgs = make_grid(val_imgs, nrow=10, scale_each=True, pad_value=10)


                if i % 1000 == 0:
                    torch.save(net.state_dict(), save_model_path + 'iter' + str(i+126000) + '.pth')
        i += 1

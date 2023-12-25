from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from layer import MultiSpectralAttentionLayer#fca

class SLC_spexy_CAE(nn.Module):
    def __init__(self):
        super(SLC_spexy_CAE, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.ReLU()
        )
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        #######################################################

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.unpooling3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.unpooling1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.encoder1(x)
        x, pos1 = self.maxpooling1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x, pos3 = self.maxpooling3(x)
        x = self.encoder4(x)

        x = self.decoder4(x)
        x = self.unpooling3(x, pos3)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.unpooling1(x, pos1)
        x = self.decoder1(x)

        return x

    def get_encoder_features(self, x):
        x = self.encoder1(x)
        #print(x.shape)torch.Size([64, 32, 28, 28])
        x, pos1 = self.maxpooling1(x)
        #print(x.shape)torch.Size([64, 32, 14, 14])
        x = self.encoder2(x)
        #print(x.shape)torch.Size([64, 64, 10, 10])
        x = self.encoder3(x)
        #print(x.shape)torch.Size([64, 64, 8, 8])
        x, pos3 = self.maxpooling3(x)
        #print(x.shape)torch.Size([64, 64, 4, 4])
        x = self.encoder4(x)
        #print(x.shape)torch.Size([64, 128, 1, 1])

        return x
class SLC_spexy_CAE_48(nn.Module):
    def __init__(self):
        super(SLC_spexy_CAE_48, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.ReLU()
        )
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=1, return_indices=True)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        #######################################################


        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.unpooling1 = nn.MaxUnpool2d(kernel_size=2, stride=1)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.encoder1(x)
        x, pos1 = self.maxpooling1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.unpooling1(x, pos1)
        x = self.decoder1(x)

        return x

    def get_encoder_features(self, x):
        x = self.encoder1(x)
        x, pos1 = self.maxpooling1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        return x
class SLC_spexy_CAE_16(nn.Module):
    def __init__(self):
        super(SLC_spexy_CAE_16, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32, momentum=0.5),
            nn.ReLU()
        )
        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpooling3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.maxpooling4 = nn.MaxPool2d(kernel_size=2, stride=1, return_indices=True)

        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )



        #######################################################
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.unpooling4 = nn.MaxUnpool2d(kernel_size=2, stride=1)
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.unpooling3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.unpooling1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        #print(x.shape)
        x = self.encoder1(x)
        #print(x.shape)
        x, pos1 = self.maxpooling1(x)
        #print(x.shape)
        x = self.encoder2(x)
        #print(x.shape)
        x = self.encoder3(x)
        #print(x.shape)
        x, pos3 = self.maxpooling3(x)
        #print(x.shape)
        x = self.encoder4(x)
        #print(x.shape)
        x, pos4 = self.maxpooling4(x)
        #print(x.shape)
        x = self.encoder5(x)
        #print(x.shape)
        x = self.decoder5(x)
        x = self.unpooling4(x,pos4)
        x = self.decoder4(x)
        x = self.unpooling3(x, pos3)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.unpooling1(x, pos1)
        x = self.decoder1(x)

        return x

    def get_encoder_features(self, x):
        x = self.encoder1(x)
        x, pos1 = self.maxpooling1(x)
        x = self.encoder2(x)
        #print(x.shape)torch.Size([64, 64, 10, 10])
        x = self.encoder3(x)
        #print(x.shape)torch.Size([64, 64, 8, 8])
        x, pos3 = self.maxpooling3(x)
        #print(x.shape)torch.Size([64, 64, 4, 4])
        x = self.encoder4(x)
        #print(x.shape)torch.Size([64, 128, 1, 1])
        x, pos4 = self.maxpooling4(x)
        # print(x.shape)
        x = self.encoder5(x)
        return x

class SLC_joint2_single(nn.Module):
    def __init__(self, num_classes=5):
        super(SLC_joint2_single, self).__init__()

        self.inplanes = 256

        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            # self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.maxpool = nn.MaxPool2d(2,2)
        self.maxpool_48 = nn.MaxPool2d(3,3)

        self.fc = nn.Linear(128 * 4, num_classes)
        # self.fca = MultiSpectralAttentionLayer(576, c2wh[576], c2wh[576], reduction=16, freq_sel_method='top16')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_spe, x_img):



        x_img = self.pre_img(x_img)
        x_spe = self.maxpool(x_spe)
        x_img = x_img / 5.859713554382324
        x_spe = x_spe / 0.18485471606254578

        x = torch.cat((x_spe, x_img), 1)
        x = self.post_slc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    def pre_img_features(self, x_img):
        x = self.pre_img(x_img)
        return x

    def pre_spe_features(self, x_spe):
        x = self.maxpool(x_spe)
        # x = x.view(x.size(0),-1)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
class SLC_joint2_single_arc(nn.Module):
    def __init__(self, num_classes=5,s=20,margin=0.4):
        super(SLC_joint2_single_arc, self).__init__()

        self.inplanes = 256

        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.maxpool = nn.MaxPool2d(2,2)
        self.maxpool_48 = nn.MaxPool2d(3,3)

        self.fc = nn.Linear(128 * 4, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

        self.s = s
        self.m = margin
        self.easy_margin = False
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # cos(pi-m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(pi-m)*m
        self.weight = Parameter(torch.FloatTensor(num_classes, 128 * 4))
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x_spe, x_img,label):
        x_img = self.pre_img(x_img)
        x_spe = self.maxpool(x_spe)

        x_img = x_img / 5.859713554382324
        x_spe = x_spe / 0.18485471606254578

        x = torch.cat((x_spe, x_img), 1)

        x = self.post_slc(x)
        x = x.view(x.size(0), -1)


        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(thta+m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda:0')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        x = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        x *= self.s
        x_test = cosine
        x_test = x_test * self.s
        return x, x_test


    def pre_img_features(self, x_img):
        x = self.pre_img(x_img)
        # x = x.view(x.size(0),-1)
        return x

    def pre_spe_features(self, x_spe):
        x = self.maxpool(x_spe)
        # x = x.view(x.size(0),-1)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
class SLC_joint2(nn.Module):
    def __init__(self, num_classes=5):
        super(SLC_joint2, self).__init__()

        self.inplanes = 576
        c2wh = dict([(64, 56), (128, 28), (256, 14), (576, 7)])
        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            # self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.maxpool = nn.MaxPool2d(2,2)
        self.maxpool_48 = nn.MaxPool2d(3,3)

        self.fc = nn.Linear(128 * 4, num_classes)
        # self.fca = MultiSpectralAttentionLayer(576, c2wh[576], c2wh[576], reduction=16, freq_sel_method='top16')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_spe_32, x_spe_48, x_spe_16, x_img):
        x_img = self.pre_img(x_img)


        x_spe_32 = self.maxpool(x_spe_32)

        x_spe_48 = self.maxpool_48(x_spe_48)

        x_spe_16 = x_spe_16

        x_img = x_img / 5.859713554382324
        x_spe_32 = x_spe_32 / 0.18485471606254578
        x_spe_48 = x_spe_48 / 0.18485471606254578
        x_spe_16 = x_spe_16 / 0.18485471606254578
        x = torch.cat((x_spe_32, x_spe_16,x_spe_48, x_img), 1)
        x = self.post_slc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    def pre_img_features(self, x_img):
        x = self.pre_img(x_img)
        return x

    def pre_spe_features(self, x_spe):
        x = self.maxpool(x_spe)
        # x = x.view(x.size(0),-1)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class SLC_muilt_all_spe(nn.Module):
    def __init__(self, num_classes=5,pyramids=[2,4,8,16]):
        super(SLC_muilt_all_spe, self).__init__()

        self.inplanes = 128*3
        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )
        self.ASPP = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2

        )
        self.FPN = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu

        )

        self.layer1 = nn.Sequential(ResNet18_TSX(5).layer1)
        self.layer2 = nn.Sequential(ResNet18_TSX(5).layer2)
        self.toplayer = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)


        self.aspp1 = ASPP_module(128, 128, pyramids[0])
        self.aspp2 = ASPP_module(128, 128, pyramids[1])
        self.aspp3 = ASPP_module(128, 128, pyramids[2])
        self.aspp4 = ASPP_module(128, 128, pyramids[3])

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            # self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.maxpool = nn.MaxPool2d(2,2)
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 4, num_classes)
        # self.fca = MultiSpectralAttentionLayer(576, c2wh[576], c2wh[576], reduction=16, freq_sel_method='top16')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    def forward(self, x_img, x_all_spe):
        # print(x_img.shape)
        x_img = self.pre_img(x_img)
        # print(x_img.shape)
        x_0 = self.FPN(x_all_spe)

        x1 = self.layer1(x_0)
        x2 = self.layer2(x1)

        p2 = self.toplayer(x2)
        p1 = self._upsample_add(p2, x1)
        p1 = self.layer2(p1)
        x_final_spe = torch.cat((x2, p1), 1)

        x_img = x_img / 5.859713554382324

        x_final_spe = x_final_spe/0.18485471606254578

        x = torch.cat((x_final_spe, x_img), 1)

        x = self.post_slc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
class SLC_joint2_arc(nn.Module):
    def __init__(self, num_classes=5,s=20,margin=0.4):
        super(SLC_joint2_arc, self).__init__()

        self.inplanes = 576

        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            # self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(128 * 4, num_classes)
        self.maxpool_48 = nn.MaxPool2d(3, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

        ##arcface
        self.s = s
        self.m = margin
        self.easy_margin = False
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # cos(pi-m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(pi-m)*m
        self.weight = Parameter(torch.FloatTensor(num_classes,  128 * 4))
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, x_spe_32, x_spe_48, x_spe_16, x_img, label):


        x_img = self.pre_img(x_img)


        x_spe_32 = self.maxpool(x_spe_32)
        x_spe_48 = self.maxpool_48(x_spe_48)

        x_spe_16 = x_spe_16

        x_img = x_img / 5.859713554382324
        x_spe_32 = x_spe_32 / 0.18485471606254578
        x_spe_48 = x_spe_48 / 0.18485471606254578
        x_spe_16 = x_spe_16 / 0.18485471606254578
        x = torch.cat((x_spe_32, x_spe_48,x_spe_16, x_img), 1)

        x = self.post_slc(x)
        x = x.view(x.size(0), -1)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(thta+m)
        if self.easy_margin:

            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(),device='cuda:0')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        x = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        x *= self.s
        x_test = cosine
        x_test = x_test*self.s
        return x,x_test
###

    def pre_img_features(self, x_img):
        x = self.pre_img(x_img)
        # x = x.view(x.size(0),-1)

        return x

    def pre_spe_features(self, x_spe):
        x = self.maxpool(x_spe)
        # x = x.view(x.size(0),-1)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class SLC_joint2_img(nn.Module):
    def __init__(self, num_classes=5):
        super(SLC_joint2_img, self).__init__()

        self.inplanes = 128

        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            # self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(128 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_img):

        x = self.pre_img(x_img)

        x = self.post_slc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class SLC_spe(nn.Module):
    def __init__(self, num_classes=5):
        super(SLC_spe, self).__init__()

        self.inplanes = 128

        self.pre_img = nn.Sequential(
            ResNet18_TSX(5).conv1,
            ResNet18_TSX(5).bn1,
            ResNet18_TSX(5).relu,
            ResNet18_TSX(5).layer1,
            ResNet18_TSX(5).layer2
        )

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            # self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(128 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x_img):

        x = self.pre_img(x_img)

        x = self.post_slc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class SLC_joint2_spe(nn.Module):
    def __init__(self, num_classes=5):
        super(SLC_joint2_spe, self).__init__()

        self.inplanes = 128

        self.post_slc = nn.Sequential(
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            self._make_layer(Bottleneck, 128, blocks=1, stride=2),
            self._make_layer(Bottleneck, 256, blocks=1, stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(256 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):

        x = self.post_slc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

""" ResNet """
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1) # for input of 64*64
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class ResNet_features(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_features, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1) # for input of 64*64


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
def ResNet18_TSX(tsx_num_class):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=tsx_num_class)
    return model



# if __name__ == '__main__':
#     net = SLC_joint()
#     print(2)

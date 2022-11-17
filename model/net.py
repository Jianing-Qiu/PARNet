import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections
from skimage import measure
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

import utils



def return_bbox_and_region(logits, y, feature_conv, weight_softmax, ratios):
    if torch.is_tensor(y):
        class_idx = y.data.cpu().numpy()
    else:
        h_x = F.softmax(logits, dim=1)
        _, idx = h_x.sort(1, True)
        class_idx = idx[:, 0].data.cpu().numpy()

    return utils.get_bbox_and_region_based_on_cam(feature_conv, weight_softmax, class_idx, ratios)


def crop_and_zoom(x_chunk, bbox):

    for i, (left, top, right, bottom) in enumerate(bbox):
        if i == 0:
            x2 = F.interpolate(x_chunk[i][:, :, top:bottom, left:right],
                               size=(224, 224), mode='bilinear', align_corners=True)
        else:
            x2 = torch.cat((x2, F.interpolate(
                x_chunk[i][:, :, top:bottom, left:right], size=(224, 224), mode='bilinear', align_corners=True)))

    return x2


def erase(x_chunk, regions):
    for i, region in enumerate(regions):

        mask = torch.ones(x_chunk[i].size()[-2], x_chunk[i].size()[-1])
        mask[region] = 0
        mask = mask[None, None, :, :]
        mask = mask.type(torch.cuda.FloatTensor)

        if i == 0:
            x2 = x_chunk[i] * mask
        else:
            x2 = torch.cat((x2, x_chunk[i] * mask))

    return x2


class PARNet(nn.Module):

    def __init__(self, num_classes, mining_times, ratios, fe_model='resnet50'):
        super(PARNet, self).__init__()

        if fe_model == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=True)
        elif fe_model == 'resnet101':
            self.feature_extractor = models.resnet101(pretrained=True)
        else:
            raise NameError('Only resnet50 and resnet101 are supported!')

        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 2048
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes)

        # self.num_parts = num_parts
        self.mining_times = mining_times
        self.ratios = ratios

        self.finalconv_name = 'layer4'
        self.feature_extractor._modules.get(
            self.finalconv_name).register_forward_hook(self.hook_conv_feature)

        self.gap_name = 'avgpool'
        self.feature_extractor._modules.get(
            self.gap_name).register_forward_hook(self.hook_gap_feature)

        self.region_net = models.resnet50(pretrained=True)
        # TODO
        # use the original avgpool
        self.region_net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # dim = 512 if resnet34
        region_num_ftrs = self.region_net.fc.in_features
        print("region_num_ftrs: ", region_num_ftrs)
        self.region_net.fc = nn.Linear(region_num_ftrs, num_classes)

        self.region_gap_name = 'avgpool'
        self.region_net._modules.get(self.region_gap_name).register_forward_hook(
            self.hook_gap_feature)

        num_of_region = self.mining_times * len(self.ratios)

        self.concat_fc = nn.Linear(
            num_ftrs + region_num_ftrs * num_of_region, num_classes)

        self.auxiliary_net = models.resnet34(pretrained=True)
        self.auxiliary_net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        aux_num_ftrs = self.auxiliary_net.fc.in_features
        self.auxiliary_net.fc = nn.Linear(aux_num_ftrs, num_classes)

        self.aux_finalconv_name = 'layer4'
        self.auxiliary_net._modules.get(
            self.aux_finalconv_name).register_forward_hook(self.aux_hook_conv_feature)

    def forward(self, x, y=None):
        # print("x: ", x)
        self.conv_feature_blobs = []

        self.gap_feature_blobs = []

        full_img_logits = self.get_logits(x)

        self.weight_softmax = np.squeeze(
            list(self.feature_extractor.parameters())[-2].data.cpu().numpy())

        self.aux_conv_feature_blobs = []
        self.aux_weight_softmax = []

        bbox_of_mined_regions = {}
        current_logits = full_img_logits
        current_x = x
        current_conv_fb = self.conv_feature_blobs[0]
        current_ws = self.weight_softmax
        erased_img_logits = {}
        for i in range(self.mining_times):
            print(">>>>>>>>>>>>>>>>>>>>>>>mining time: ", i)
            bbox_dict, region_dict = return_bbox_and_region(
                current_logits, y, current_conv_fb, current_ws, self.ratios)
            bbox_of_mined_regions[i] = bbox_dict

            if i == self.mining_times - 1:
                break

            for _, r in region_dict.items():
                erased_x = erase(torch.chunk(
                    current_x, current_x.size()[0]), r)
                current_x = erased_x

            erased_img_logits[i] = self.auxiliary_net(erased_x)

            current_logits = erased_img_logits[i]
            current_conv_fb = self.aux_conv_feature_blobs[-1]
            current_ws = np.squeeze(
                list(self.auxiliary_net.parameters())[-2].data.cpu().numpy())
            self.aux_weight_softmax.append(current_ws)

        x_chunk = torch.chunk(x, x.size()[0])

        regions_logits = {}
        for mining_time, bbox_dict in bbox_of_mined_regions.items():
            regions_logits_sub = {}
            for ratio, bbox in bbox_dict.items():
                regions_logits_sub[ratio] = self.region_net(
                    crop_and_zoom(x_chunk, bbox))
            regions_logits[mining_time] = regions_logits_sub
        print(">>>>>>>>>>>>>>>>>>>>the length of regions_logits: ",
              len(regions_logits))

        concat_features = torch.cat(self.gap_feature_blobs, dim=1)
        # print("concat_features size: ", concat_features.size())

        concat_logits = self.concat_fc(concat_features)

        return full_img_logits, regions_logits, concat_logits, erased_img_logits

    def hook_conv_feature(self, module, input, output):
        self.conv_feature_blobs.append(output.data.cpu().numpy())

    def aux_hook_conv_feature(self, module, input, output):
        self.aux_conv_feature_blobs.append(output.data.cpu().numpy())

    def hook_gap_feature(self, module, input, output):
        if output.data.size()[0] == 1:
            gf = output.data.squeeze(3)
            gf = gf.squeeze(2)
            self.gap_feature_blobs.append(gf)
        else:
            self.gap_feature_blobs.append(output.data.squeeze())

    def get_logits(self, x):
        return self.feature_extractor(x)


class BaselineNet(nn.Module):

    def __init__(self, num_classes, fe_model='resnet50'):
        super(BaselineNet, self).__init__()

        if fe_model == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=True)
        elif fe_model == 'resnet101':
            self.feature_extractor = models.resnet101(pretrained=True)
        else:
            raise NameError('Only resnet50 and resnet101 are supported!')

        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.feature_extractor(x)

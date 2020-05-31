# -*- coding:utf-8 -*-
# time:2020/5/20 8:43 PM
# author:ZhaoH
import torch
import torch.nn as nn
from torchvision import models


def set_parameter_requires_grad(model, requires_grad):
    if requires_grad:
        for param in model.parameters():
            param.requires_grad = False


class ResnetExtractor(nn.Module):
    def __init__(self, out_dim, use_pretrained=True,  requires_grad=False):
        super(ResnetExtractor, self).__init__()
        self.model_resnet50 = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(self.model_resnet50, requires_grad)
        self.num_ftrs = self.model_resnet50.fc.in_features
        self.model_fc_video = nn.Sequential(
            nn.Linear(self.num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )
        self.model_fc_bgm = nn.Sequential(
            nn.Linear(self.num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )

    def forward(self, input_video, input_bgm):
        x_video = self.model_resnet50(input_video)
        x_video = self.model_fc_video(x_video)

        x_bgm = self.model_resnet50(input_bgm)
        x_bgm = self.model_fc_bgm(x_bgm)

        return x_video, x_bgm


class VMBModel(nn.Module):
    def __init__(self, pos_nums, fea_dim, out_dim, res_use_pretrained=True,  res_requires_grad=False):
        super(VMBModel, self).__init__()
        self.pos_nums = pos_nums
        self.fea_dim = fea_dim
        self.out_dim = out_dim

        """
            init feature extractor layer
        """
        self.model_resnet50 = ResnetExtractor(self.fea_dim, res_use_pretrained, res_requires_grad)

        """
            init beat merge layer
        """
        self.video_att_layer = nn.Linear(self.fea_dim * self.pos_nums, self.pos_nums)
        self.video_att_activation = nn.Softmax(dim=1)
        self.video_att_bn1d = nn.BatchNorm1d(self.pos_nums)
        self.bgm_att_layer = nn.Linear(self.fea_dim * self.pos_nums, self.pos_nums)
        self.bgm_att_activation = nn.Softmax(dim=1)
        self.bgm_att_bn1d = nn.BatchNorm1d(self.pos_nums)


        """
            init beat encoding layer
        """
        self.video_fc = nn.Linear(self.pos_nums, self.pos_nums)
        self.video_fc_activation = nn.ReLU()
        self.video_bn1d = nn.BatchNorm1d(self.pos_nums)
        self.bgm_fc = nn.Linear(self.pos_nums, self.pos_nums)
        self.bgm_fc_activation = nn.ReLU()
        self.bgm_bn1d = nn.BatchNorm1d(self.pos_nums)

        """
            init tower layer
        """
        self.video_tower_fc = nn.Linear(self.fea_dim, self.out_dim)
        self.bgm_tower_fc = nn.Linear(self.fea_dim, self.out_dim)

    def forward(self, input_video, input_bgm):
        video_fea = []
        bgm_fea = []

        """
           feature extractor part with resnet
        """
        for tmp_v, tmp_b in zip(input_video, input_bgm):
            tmp_v_out, tmp_b_out = self.model_resnet50(tmp_v, tmp_b)
            video_fea.append(tmp_v_out)
            bgm_fea.append(tmp_b_out)

        """
           beat merge part
        """
        video_fea_t = torch.tensor(video_fea, dtype=torch.float32).view(-1)
        bgm_fea_t = torch.tensor(bgm_fea, dtype=torch.float32).view(-1)

        video_att = self.video_att_activation(self.video_att_layer(video_fea_t))
        bgm_att = self.bgm_att_activation(self.bgm_att_layer(bgm_fea_t))

        video_fea_merge = 0
        bgm_fea_merge = 0
        for ind in range(self.pos_nums):
            video_fea_merge += video_fea[ind] * video_att[:, ind].view((video_att.size()[0], 1))
            bgm_fea_merge += bgm_fea[ind] * bgm_att[:, ind].view((bgm_att.size()[0], 1))

        video_fea_merge = self.video_att_bn1d(video_fea_merge)
        bgm_fea_merge = self.bgm_att_bn1d(bgm_fea_merge)

        """
           beat encoding part
        """
        video_att_encoding = self.video_bn1d(self.video_fc_activation(self.video_fc(video_att)))
        bgm_att_encoding = self.bgm_bn1d(self.bgm_fc_activation(self.bgm_fc(bgm_att)))

        """
           tower part
        """
        video_tower_out = self.video_tower_fc(video_fea_merge)
        bgm_tower_out = self.bgm_tower_fc(bgm_fea_merge)

        return video_att_encoding, bgm_att_encoding, video_tower_out, bgm_tower_out



























# -*- coding:utf-8 -*-
# time:2020/5/31 10:09 AM
# author:ZhaoH
import numpy as np
import torch

from data_prepare import video_prepare, bgm_prepare
from VMBModel import VMBModel


def model_train(video_file, bgm_file):

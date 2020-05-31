# -*- coding:utf-8 -*-
# time:2020/5/23 2:49 PM
# author:ZhaoH
import cv2
import numpy as np
import librosa


def video_prepare(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    res = []
    count = 0
    while success:
        if count % fps == 0:
            res.append(image)
        print('Process %dth seconds: ' % int(count / fps), success)
        success, image = vidcap.read()
        count += 1

    return res


def bgm_prepare(bgm_path, config):
    data, _ = librosa.core.load(bgm_path, sr=config.sampling_rate, res_type="kaiser_fast")
    res = []
    max_len = len(data)
    for idx in range(60):
        if idx * config.sampling_rate <= max_len:
            pre_idx = idx * config.sampling_rate - config.sampling_rate / 2
            post_idx = idx * config.sampling_rate + config.sampling_rate / 2
            pre_idx = pre_idx if pre_idx >= 0 else 0
            post_idx = post_idx if post_idx < max_len else max_len - 1
            idx_part = data[pre_idx : post_idx + 1]
            idx_part = np.pad(idx_part,
                              (-pre_idx if pre_idx < 0 else 0,
                               post_idx - max_len + 1 if (post_idx - max_len + 1) > 0 else 0),
                              "constant", constant_values=(0, 0))
            idx_melspec = librosa.feature.melspectrogram(idx_part, sr=config.sampling_rate, n_mels=200)
            idx_logspec = librosa.core.power_to_db(idx_melspec)
            idx_logspec = np.expand_dims(idx_logspec, axis=-1)
            res.append(idx_logspec)

        else:
            res.append(np.zeros(shape=(config.dim[0], config.dim[1], 1)))

    return res


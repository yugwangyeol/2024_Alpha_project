import os 
from numpy.random import f, permutation, rand
from PIL import Image
import time
import torch
import random
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import cv2

class VideoAnomalyDataset_C3D(Dataset):
    """Video Anomaly Dataset for DAD_jigsaw without object detection."""
    def __init__(self,
                 data_dir, 
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        
        self.data_dir = data_dir
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0
        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

        if 'train' in data_dir:
            self.test_stage = False
        elif 'test' in data_dir:
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'
        self.sample_step = 5 if not self.test_stage else 1

        self.frames_list = []
        self._load_data(file_list)

    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1    
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            frames = os.listdir(os.path.join(self.data_dir, video_file))
            frames.sort()
            self.videos += 1
            length = len(frames)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step):
                self.frames_list.append({"video_name": video_file, "frame": frame})

        print("Load {} videos {} frames in {} s.".format(self.videos, total_frames, time.time() - t0))

    def __len__(self):
        return len(self.frames_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx): 
        temporal_flag = idx % 2 == 0 
        record = self.frames_list[idx]
        if self.test_stage:
            perm = np.arange(self.frame_num)
        else:
            if random.random() < 0.0001:
                perm = np.arange(self.frame_num)
            else:
                perm = np.random.permutation(self.frame_num)
        clip = self.get_clip(record["video_name"], record["frame"])

        if not temporal_flag and not self.test_stage:
            if random.random() < 0.0001:
                spatial_perm = np.arange(9)
            else:
                spatial_perm = np.random.permutation(9)
        else:
            spatial_perm = np.arange(9)
        clip = self.jigsaw(clip, border=2, patch_size=20, permutation=spatial_perm, dropout=False)
        clip = torch.from_numpy(clip)

        # NOT permute clips containing static contents
        if (clip[:, -1, :, :] - clip[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(self.frame_num)

        if temporal_flag:
            clip = clip[:, perm, :, :]
        clip = torch.clamp(clip, 0., 1.)

        ret = {"video": record["video_name"], "frame": record["frame"], "clip": clip, "label": perm, 
               "trans_label": spatial_perm, "temporal": temporal_flag}
        return ret   

    def get_clip(self, video_name, frame):
        """
        Reads a sequence of frames centered around the specified frame.
        Returns the entire frame as an object.
        """
        video_dir = os.path.join(self.data_dir, video_name)
        frame_list = os.listdir(video_dir)
        img = self.read_frame_data(video_dir, frame, frame_list)
        return img

    def read_single_frame(self, video_dir, frame, frame_list):
        
        transform = transforms.ToTensor()
        img = None

        frame_ = "img_{}.png".format(frame)

        assert (frame_ in frame_list),\
            "The frame {} is out of the range:{}.".format(int(frame_), len(frame_list))

        png_dir = '{}/{}'.format(video_dir, frame_)
        assert os.path.exists(png_dir), "{} isn\'t exists.".format(png_dir)

        img = Image.open(png_dir)
        img = transform(img).unsqueeze(dim=0) 
        img = img.permute([1, 0, 2, 3])
        return img

    def read_frame_data(self, video_dir, frame, frame_list):
        img = None
        for f in range(self.frame_num):
            if frame + f < len(frame_list):
                _img = self.read_single_frame(video_dir, frame + f, frame_list)
                if f == 0:
                    img = _img
                else:
                    img = torch.cat((img, _img), dim=1)
            else:
                break
        return img

    def split_image(self, clip, border=2, patch_size=20):
        """
        clip: (C, T, H, W)
        """
        patch_list = []
        for i in range(3):
            for j in range(3):
                y_offset = border + patch_size * i
                x_offset = border + patch_size * j
                patch_list.append(clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size])
        return patch_list

    def concat(self, patch_list, border=2, patch_size=20, permutation=np.arange(9), num=3, dropout=False):
        """
        patch_list: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)
        drop_ind = random.randint(0, len(permutation) - 1)
        for p_ind, i in enumerate(permutation):
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
        return clip

    def jigsaw(self, clip, border=2, patch_size=20, permutation=None, dropout=False):
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permutation=permutation, num=3, dropout=dropout)
        return clip

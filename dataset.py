import os 
from PIL import Image
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class VideoAnomalyDataset_C3D(Dataset):
    """Video Anomaly Dataset for DAD_Jigsaw without object detection."""
    def __init__(self,
                 data_dir, 
                 frame_num=7,
                 static_threshold=0.1):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        
        self.data_dir = data_dir
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []
        self.phase = 'testing' if 'test' in data_dir else 'training'
        self.sample_step = 5 if self.phase == 'training' else 1

        self.frames_list = []
        self._load_data(file_list)

    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        start_ind = self.half_frame_num if self.phase == 'testing' else self.frame_num - 1    
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            frames = os.listdir(os.path.join(self.data_dir, video_file))
            frames.sort()
            length = len(frames)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step):
                self.frames_list.append({"video_name": video_file, "frame": frame})

        print("Loaded {} videos with {} frames in {:.2f} seconds.".format(len(self.videos_list), total_frames, time.time() - t0))

    def __len__(self):
        return len(self.frames_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx): 
        temporal_flag = idx % 2 == 0
        record = self.frames_list[idx]
        if self.phase == 'testing':
            perm = np.arange(self.frame_num)
        else:
            perm = np.random.permutation(self.frame_num) if random.random() >= 0.0001 else np.arange(self.frame_num)

        # 전체 프레임을 객체로 간주하고 읽어오기
        clip = self.get_clip(record["video_name"], record["frame"])

        if not temporal_flag and self.phase == 'training':
            spatial_perm = np.random.permutation(9) if random.random() >= 0.0001 else np.arange(9)
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
        Returns the entire frame as a clip.
        """
        video_dir = os.path.join(self.data_dir, video_name)
        frame_list = os.listdir(video_dir)

        frame_list.sort()  # Ensure frame ordering is correct

        img_list = []
        for i in range(frame - self.half_frame_num, frame + self.half_frame_num + 1):
            img_path = os.path.join(video_dir, f"img_{i}.png")
            img = Image.open(img_path).convert('L')  # Load as 1-channel grayscale
            img = np.array(img)

            # Convert 1-channel to 3-channel by replicating
            img = np.stack([img] * 3, axis=0)  # Shape: (3, H, W)
            img_list.append(img)

        clip = np.array(img_list).transpose(1, 0, 2, 3)  # (C, T, H, W)
        return clip

    def split_image(self, clip, border=2, patch_size=20):
        """
        Splits the clip into patches.
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
        Concatenate the patches back into a full image.
        patch_list: [(C, T, h1, w1)]
        """
        clip = np.zeros((3, self.frame_num, 64, 64), dtype=np.float32)  # (C, T, H, W)
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
        """
        Applies jigsaw permutation to the clip.
        """
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permutation=permutation, num=3, dropout=dropout)
        return clip

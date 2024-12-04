import os 
from PIL import Image
import time
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class VideoAnomalyDataset_Clip(Dataset):
    def __init__(self,
                data_dir,
                sample_num=7,    
                num_clips=5,      
                border=2,
                patch_size=20,
                static_threshold=0.1):
                
        assert os.path.exists(data_dir), f"{data_dir} does not exist."
        assert sample_num % 2 == 1, 'We prefer odd number of frames'
        
        self.data_dir = data_dir
        self.sample_num = sample_num
        self.half_sample_num = sample_num // 2 
        self.num_clips = num_clips
        self.border = border
        self.patch_size = patch_size  
        self.static_threshold = static_threshold
        
        self.videos_list = []
        self.clips_list = []
        self.phase = 'testing' if 'test' in data_dir else 'training'
        
        file_list = os.listdir(data_dir)
        file_list.sort()
        self._load_data(file_list)

    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
                
            frames = os.listdir(os.path.join(self.data_dir, video_file))
            frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            length = len(frames)
            total_frames += length

            # 각 클립의 중심 프레임 위치 계산
            center_frame = self.half_sample_num
            while center_frame + (self.sample_num * (self.num_clips-1)) + self.half_sample_num < length:
                self.clips_list.append({
                    "video_name": video_file,
                    "center_frame": center_frame
                })
                
                if self.phase == 'training':
                    center_frame += 5
                else:
                    center_frame += 1
                    
        print(f"{self.phase} Loaded {len(self.videos_list)} videos with {total_frames} frames "
              f"and {len(self.clips_list)} clips in {time.time() - t0:.2f} seconds.")

    def get_cube(self, clips, patch_idx):
        """모든 클립에서 동일한 위치의 패치를 추출하여 큐브로 만듦"""
        y = patch_idx // 3
        x = patch_idx % 3
        y_offset = self.border + self.patch_size * y
        x_offset = self.border + self.patch_size * x
        return clips[:, :, y_offset:y_offset + self.patch_size, x_offset:x_offset + self.patch_size]

    def spatial_permute_cubes(self, clips):
        """모든 클립에 대해 동일한 큐브 위치 변경을 적용"""
        cubes = []
        for i in range(9):  # 3x3 패치 위치
            cube = self.get_cube(clips, i)
            cubes.append(cube)
            
        spatial_perm = np.random.permutation(9) if random.random() >= 0.0001 else np.arange(9)
        new_clips = np.zeros_like(clips)
        
        for new_idx, old_idx in enumerate(spatial_perm):
            y = new_idx // 3
            x = new_idx % 3
            y_offset = self.border + self.patch_size * y
            x_offset = self.border + self.patch_size * x
            new_clips[:, :, y_offset:y_offset + self.patch_size, x_offset:x_offset + self.patch_size] = cubes[old_idx]
            
        return new_clips, spatial_perm

    def get_clip(self, video_name, center_frame):
        """중심 프레임을 기준으로 클립 로드"""
        video_dir = os.path.join(self.data_dir, video_name)
        frame_list = os.listdir(video_dir)
        frame_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        img_list = []
        
        for i in range(center_frame - self.half_sample_num, 
                      center_frame + self.half_sample_num + 1):
            img_path = os.path.join(video_dir, f"img_{i}.png")
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize((64, 64))
                img = np.array(img) / 255.0
                img = np.array(img, dtype=np.float32)
                img_list.append(img)
            except:
                print(f"Error loading frame: {img_path}")
                return None
                
        if len(img_list) != self.sample_num:
            return None
            
        arr_expanded = np.expand_dims(np.array(img_list), axis=1)
        clip = arr_expanded.transpose(1, 0, 2, 3)
        return clip

    def __getitem__(self, idx):
        record = self.clips_list[idx]
        video_name = record["video_name"]
        center_frame = record["center_frame"]
        
        # 각 클립의 중심 프레임을 기준으로 클립 로드
        clips_list = []
        for i in range(self.num_clips):
            clip_center = center_frame + i * self.sample_num
            clip = self.get_clip(video_name, clip_center)
            if clip is None:
                raise ValueError(f"Failed to load clip from {video_name} at frame {clip_center}")
            clips_list.append(clip)
            
        # 정적 체크 - 연속된 클립 간의 차이를 확인
        clips = np.concatenate(clips_list, axis=1)
        is_static = (clips[:, -1, :, :] - clips[:, 0, :, :]).max() < self.static_threshold
        

        # temporal 및 spatial 변환 적용
        if self.phase == 'training' and not is_static:
            temp_perm = np.random.permutation(self.num_clips)
            clips_list = [clips_list[i] for i in temp_perm]
            clips = np.concatenate(clips_list, axis=1)
            clips, spatial_perm = self.spatial_permute_cubes(clips)
        else:
            temp_perm = np.arange(self.num_clips)
            spatial_perm = np.arange(9)
            clips = np.concatenate(clips_list, axis=1)

        clips = torch.from_numpy(clips).float()
        clips = torch.clamp(clips, 0., 1.)

        return {
            "video": video_name,
            "center_frame": center_frame,
            "clips": clips,
            "label": temp_perm,
            "spatial_label": spatial_perm,
            "temporal": not is_static 
        }

    def __len__(self):
        return len(self.clips_list)
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
                 frame_num=7, # 하나의 비디오 클립에 들어갈 프레임 수
                 static_threshold=0.1): #영상 클립이 정적(변화가 없는)인지 판단할 때 사용하는 임계값

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        
        self.data_dir = data_dir
        self.static_threshold = static_threshold
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'We prefer odd number of frames' # 프렘이 수가 홀수 인지 아닌지 판단
        self.half_frame_num = self.frame_num // 2 
        # 중심 프레임을 기준으로 양옆의 프레임 수를 계산
        #예를 들어, frame_num=7일 때, 중심 프레임 앞뒤로 3개의 프레임을 가짐

        self.videos_list = [] 
        self.phase = 'testing' if 'test' in data_dir else 'training'
        self.sample_step = 5 if self.phase == 'training' else 1 
        # 데이터를 샘플링할 때 사용할 스텝 크기를 설정
        # training일 때는 5 프레임마다 샘플링하고, testing일 때는 1 프레임씩 샘플링

        self.frames_list = []
        self._load_data(file_list)

    def _load_data(self, file_list):
        t0 = time.time() # 데이터 로드 시간
        total_frames = 0
        start_ind = self.half_frame_num if self.phase == 'testing' else self.frame_num - 1
        # 테스트 단계에서는 중심 프레임(half_frame_num)을 기준으로 데이터를 불러오고,
        # 학습 단계에서는 frame_num - 1만큼 이동하여 시작 프레임을 설정
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file) # 없는 파일 video_list에 추가
            frames = os.listdir(os.path.join(self.data_dir, video_file))
            frames.sort()
            length = len(frames)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step): 
                # 비디오 프레임들을 일정 스텝(sample_step)으로 순회하면서 frame_list에 프레임 정보를 저장
                self.frames_list.append({"video_name": video_file, "frame": frame})
                # frame_list에 비디오 이름과 프레임 번호를 저장

        print("{} Loaded {} videos with {} frames in {:.2f} seconds.".format(self.phase ,len(self.videos_list), total_frames, time.time() - t0))

    def __len__(self):
        return len(self.frames_list)

    def __video_list__(self):
        return self.videos_list

    def __getitem__(self, idx): 
        temporal_flag = idx % 2 == 0
        # 짝수 인덱스일 때 temporal_flag를 True로 설정하여 클립의 시계적 순서를 그대로 유지하게 됨
        record = self.frames_list[idx]
        # 인덱스에 해당하는 비디오 프레임 정보를 가져옴

        if self.phase == 'testing':
            perm = np.arange(self.frame_num)
        else:
            perm = np.random.permutation(self.frame_num) if random.random() >= 0.0001 else np.arange(self.frame_num)
            # 학습 단계에서, 랜덤한 순서로 프레임을 섞음
            # 테스트 단계에서는 프레임 순서를 그대로 유지

        clip = self.get_clip(record["video_name"], record["frame"])
        # 비디오의 특정 프레임을 기준으로 여러 프레임을 읽어와 클립(Clip) 형태로 가져옴
        # (C, T, H, W) 형식의 4차원 텐서로 반환
        # C (채널): 이미지의 채널 수. 이 코드에서는 3채널(컬러 이미지)로 가정
        # T (타임/프레임): 클립에 포함된 프레임 수 (self.frame_num)
        # H (높이) 및 W (너비): 각 프레임의 높이와 너비

        if clip is None: # 수정
            # 에러 처리 로직 추가
            print(f"Error getting clip for {record['video_name']} at frame {record['frame']}")
            # 대체 클립 반환 또는 예외 처리
            
        # NaN 체크 추가 
        if np.isnan(clip).any():# 수정
            print(f"NaN detected in clip: {record['video_name']} at frame {record['frame']}")

        if not temporal_flag and self.phase == 'training':
            spatial_perm = np.random.permutation(9) if random.random() >= 0.0001 else np.arange(9)
            # spatial_perm: 9개의 패치(3x3 그리드)를 가리키는 순서를 랜덤하게 섞어 반환하거나 원래 순서(np.arange(9))를 반환
        else:
            spatial_perm = np.arange(9)

        clip = self.jigsaw(clip, border=2, patch_size=20, permutation=spatial_perm, dropout=False)
        # 클립을 조각낸 뒤 (jigsaw 메서드를 호출) 다시 결합
        # 이때 border=2 및 patch_size=20을 사용하여 64x64 크기의 이미지를 3x3의 조각으로 나눕
        clip = torch.from_numpy(clip)

        # NOT permute clips containing static contents
        if (clip[:, -1, :, :] - clip[:, 0, :, :]).abs().max() < self.static_threshold:
            perm = np.arange(self.frame_num)
        # 정적 클립을 판단하는 조건
        # 클립의 마지막 프레임과 첫 번째 프레임 간의 픽셀 값 차이가 static_threshold 이하일 때 perm을 원래 순서로 설정하여, 정적인 영상에서는 순서 섞기를 적용하지 않음

        if temporal_flag:
            clip = clip[:, perm, :, :]
        # temporal_flag가 True인 경우, 클립을 perm 순서대로 재배치함 -> 즉, 랜덤 순서 섞기가 적용
        clip = torch.clamp(clip, 0., 1.)
        # 클립의 픽셀 값이 0과 1 사이로 클램핑 -> 이는 픽셀 값이 잘못된 범위로 벗어나는 경우를 방지하기 위해 사용

        ret = {"video": record["video_name"], "frame": record["frame"], "clip": clip, "label": perm, 
               "trans_label": spatial_perm, "temporal": temporal_flag}
        
        # Video: 비디오 이름.
        # frame: 해당 클립의 기준이 되는 프레임 번호.
        # clip: 실제로 모델이 사용할 클립 데이터 (이미지 텐서).
        # label: 클립의 시간 순서를 나타내는 라벨.
        # trans_label: 클립의 공간 순서를 나타내는 라벨.
        # temporal: 클립이 시간 순서 섞기가 적용되었는지 여부
        return ret

    """def get_clip(self, video_name, frame):

        video_dir = os.path.join(self.data_dir, video_name)
        frame_list = os.listdir(video_dir)
        frame_list.sort()  # Ensure frame ordering is correct

        img_list = []
        for i in range(frame - self.half_frame_num, frame + self.half_frame_num + 1):
            #print(f"img_{i}.png")
            img_path = os.path.join(video_dir, f"img_{i}.png")
            img = Image.open(img_path).convert('L')  # Load as 1-channel grayscale
            img = np.array(img)

            # Convert 1-channel to 3-channel by replicating
            #img = np.stack([img] * 3, axis=0)  # Shape: (3, H, W)
            # 1채널(그레이스케일)을 3채널로 복사하여 (3, H, W) 형식의 배열로 만듭니다.
            img_list.append(img)
        arr_expanded = np.expand_dims(np.array(img_list), axis=1)
        clip = arr_expanded.transpose(1, 0, 2, 3)  # (C, T, H, W)
        # 프레임 리스트를 하나의 클립으로 결합하여 (C, T, H, W) 형식의 4차원 배열로 반환
        return clip"""

    def get_clip(self, video_name, frame): # 수정
        video_dir = os.path.join(self.data_dir, video_name)
        frame_list = os.listdir(video_dir)
        # 수정된 정렬 방식
        frame_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        img_list = []
        for i in range(frame - self.half_frame_num, frame + self.half_frame_num + 1):
            img_path = os.path.join(video_dir, f"img_{i}.png")
            try:
                img = Image.open(img_path).convert('L')
                img = np.array(img) / 255.0  # 정규화 추가
                img_list.append(img)
            except:
                print(f"Error loading frame: {img_path}")
                return None
                
        if len(img_list) != self.frame_num:
            print(f"Incomplete clip for {video_name} at frame {frame}")
            return None
            
        arr_expanded = np.expand_dims(np.array(img_list), axis=1)
        clip = arr_expanded.transpose(1, 0, 2, 3)
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
        clip = np.zeros((1, self.frame_num, 64, 64), dtype=np.float32)  # (C, T, H, W)
        # 64x64 크기의 빈 클립을 생성
        drop_ind = random.randint(0, len(permutation) - 1)
        for p_ind, i in enumerate(permutation): #주어진 순서(permutation)대로 패치를 배치
            if drop_ind == p_ind and dropout:
                continue
            y = i // num
            x = i % num
            y_offset = border + patch_size * y
            x_offset = border + patch_size * x
            clip[:, :, y_offset: y_offset + patch_size, x_offset: x_offset + patch_size] = patch_list[p_ind]
            # 각 패치를 정해진 위치에 맞춰 클립에 삽입
        return clip

    def jigsaw(self, clip, border=2, patch_size=20, permutation=None, dropout=False):
        """
        Applies jigsaw permutation to the clip.
        """
        patch_list = self.split_image(clip, border, patch_size)
        clip = self.concat(patch_list, border=border, patch_size=patch_size, permutation=permutation, num=3, dropout=dropout)
        return clip

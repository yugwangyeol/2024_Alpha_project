import os
from PIL import Image
import time
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

class VideoAnomalyDataset(Dataset):
    def __init__(self,
                 data_dir=None, 
                 dataset='shanghaitech',
                 frame_num=7):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        
        self.dataset = dataset
        self.data_dir = data_dir

        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0
        self.frame_num = frame_num 
        assert self.frame_num % 2 == 1, 'Frame number should be an odd number for centered cropping'
        self.half_frame_num = self.frame_num // 2  

        self.videos_list = []

        if 'train' in data_dir:
            self.test_stage = False
        elif 'test' in data_dir:
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'

        self.sample_step = 1 if self.test_stage else 5

        self.objects_list = []
        self._load_data(file_list)
        self.save_objects()
    
    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1    

        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)

            frame_list = os.listdir(self.data_dir + '/' + video_file)
            self.videos += 1
            length = len(frame_list)
            total_frames += length

            for frame in range(start_ind, length - start_ind, self.sample_step):
                self.objects_list.append({"video_name": video_file, "frame": frame})

        print("Loaded {} videos with {} frames in total, in {} s.".format(self.videos, total_frames, time.time() - t0))

    def save_objects(self):
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)

        for i in tqdm(range(len(self.objects_list))):
            record = self.objects_list[i]
            video_dir = os.path.join(self.dataset, self.phase, record["video_name"])

            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            obj = self.get_object(record["video_name"], record["frame"]).numpy()
            np.save(os.path.join(video_dir, str(record['frame']) + '_full_frame.npy'), obj)

    def __len__(self):
        return len(self.objects_list)

    def __video_list__(self):
        return self.videos_list

    def get_object(self, video_name, frame):
        frame_data = self.get_frame(video_name, frame)
        obj = frame_data  
        return obj

    def get_frame(self, video_name, frame):
        video_dir = self.data_dir + '/' + video_name + '/'
        frame_list = os.listdir(video_dir)

        img = self.read_frame_data(video_dir, frame, frame_list)
        return img

    def read_single_frame(self, video_dir, frame, frame_list):
        transform = transforms.ToTensor()  
        img = None

        frame_ = "{:04d}.png".format(frame)  
        assert (frame_ in frame_list), "The frame {} is out of the range:{}.".format(frame_, len(frame_list))
        
        # 이미지 읽기
        jpg_dir = '{}/{}'.format(video_dir, frame_)
        assert os.path.exists(jpg_dir), "{} doesn't exist.".format(jpg_dir)

        img = Image.open(jpg_dir)
        img = transform(img).unsqueeze(dim=0)  
        img = img.permute([1, 0, 2, 3])  
        return img

    def read_frame_data(self, video_dir, frame, frame_list):
        img = None
        for f in range(self.frame_num):
            _img = self.read_single_frame(video_dir, frame + f, frame_list)
            if f == 0:
                img = _img
            else:
                img = torch.cat((img, _img), dim=1) 
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="patch generation")
    parser.add_argument("--dataset", type=str, default='shanghaitech')
    parser.add_argument("--phase", type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--filter_ratio", type=float, default=0.8)
    parser.add_argument("--sample_num", type=int, default=9)

    args = parser.parse_args()
    data_dir = "/home/work/Alpha/Jigsaw-VAD/"  # 데이터셋의 루트 경로
    shanghai_dataset = VideoAnomalyDataset(data_dir=data_dir + args.dataset + '/' + args.phase + 'ing/front_IR/frames', 
                                           dataset=args.dataset,
                                           frame_num=args.sample_num)

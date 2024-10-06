import os
from PIL import Image
import time
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoAnomalyDataset(Dataset):
    """Video Anomaly Dataset for DAD_Jigsaw."""
    def __init__(self,
                 data_dir=None, 
                 frame_num=7,
                 data_type=None):

        assert os.path.exists(data_dir), "{} does not exist.".format(data_dir)
        
        self.data_dir = data_dir
        self.data_type = data_type
        file_list = os.listdir(data_dir)
        file_list.sort()

        self.videos = 0

        self.frame_num = frame_num
        assert self.frame_num % 2 == 1, 'odd number is preferred'
        self.half_frame_num = self.frame_num // 2

        self.videos_list = []

        self.cache_clip = None 
        self.cache_video = None
        self.cache_frame = None

        if 'train' in data_dir:
            self.test_stage = False
        elif 'test' in data_dir:
            self.test_stage = True
        else:
            raise ValueError("data dir: {} is error, not train or test.".format(data_dir))

        self.phase = 'testing' if self.test_stage else 'training'

        self.sample_step = 1 if self.test_stage else 5

        self.frames_list = []
        self._load_data(file_list)
        self.save_frames()
    
    def _load_data(self, file_list):
        t0 = time.time()
        total_frames = 0
        start_ind = self.half_frame_num if self.test_stage else self.frame_num - 1    
        for video_file in file_list:
            if video_file not in self.videos_list:
                self.videos_list.append(video_file)
            l = os.listdir(self.data_dir + '/' + video_file)
            self.videos += 1
            length = len(l)
            total_frames += length
            for frame in range(start_ind, length - start_ind, self.sample_step):
                self.frames_list.append({"video_name":video_file, "frame":frame})
        print("Load {} videos {} frames, in {} s.".format(self.videos, total_frames, time.time() - t0))

    def save_frames(self):
        if not os.path.exists('DAD_Jigsaw'):
            os.makedirs('DAD_Jigsaw')
        for i in tqdm(range(len(self.frames_list))):
            record = self.frames_list[i]
            frame = self.get_frame(record["video_name"], record["frame"])
            video_dir = os.path.join('DAD_Jigsaw', self.phase, self.data_type, record["video_name"])
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            frame = frame.numpy()
            np.save(os.path.join(video_dir, str(record['frame']) + '.npy'), frame)
        

    def __len__(self):
        return len(self.frames_list)

    def __video_list__(self):
        return self.videos_list

    def get_frame(self, video_name, frame):
        video_dir = self.data_dir + '/' + video_name + '/'
        frame_list = os.listdir(video_dir)
        img = self.read_frame_data(video_dir, frame, frame_list)
        return img

    def read_single_frame(self, video_dir, frame, frame_list):
        from torchvision import transforms
        transform = transforms.ToTensor()

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
                print(f"Warning: Frame {frame + f} exceeds the available frame range in {video_dir}")
                break
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="patch generation")
    parser.add_argument("--phase", type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--sample_num", type=int, default=9)
    parser.add_argument("--data_type", type=str, default='top_IR', 
                        choices=['front_depth', 'front_IR', 'top_depth', 'top_IR'])
    parser.add_argument("--data_dir", type=str, default="/home/work/Alpha/Jigsaw-VAD")

    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, 'DAD_Jigsaw', args.phase + 'ing', args.data_type, 'frames')
    dataset = VideoAnomalyDataset(data_dir=data_dir, 
                                  frame_num=args.sample_num,
                                  data_type=args.data_type)
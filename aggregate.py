import pickle
import os
import numpy as np
from scipy.ndimage.measurements import label
from tool import evaluate
import argparse
from scipy.ndimage import convolve
import torch.nn.functional as F
import torch
import math


def video_label_length(dataset='DAD_Jigsaw'):
    label_path = "../DAD_Jigsaw/testing/frame_masks"
    video_length = {}
    files = sorted(os.listdir(label_path))
    length = 0
    for f in files:
        label = np.load("{}/{}".format(label_path, f))
        video_length[f.split(".")[0]] = label.shape[0] # 각 파일을 읽고, 그 파일의 프레임 길이를 video_length 딕셔너리에 저장
        length += label.shape[0] # label.shape[0]는 해당 파일에 있는 프레임 수를 의미
    return video_length


def score_smoothing(score, ws=43, function='mean', sigma=10):
    assert ws % 2 == 1, 'window size must be odd' # 윈도우 크기 ws가 홀수여야 한다는 것을 보장
    assert function in ['mean', 'gaussian'], 'wrong type of window function'

    r = ws // 2
    weight = np.ones(ws) # 윈도우 크기의 중앙 위치를 r에 저장하고, weight라는 크기 ws의 배열을 모두 1로 초기화
    for i in range(ws):
        if function == 'mean':
            weight[i] = 1. / ws
        elif function == 'gaussian':
            weight[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    weight /= weight.sum()
    new_score = score.copy()
    new_score[r: score.shape[0] - r] = np.correlate(score, weight, mode='valid')
    # 점수 score 배열에 가중치 weight를 사용하여 이동 평균을 계산하고, new_score에 저장함
    # 이 작업은 np.correlate 함수를 사용하여 수행
    return new_score


def load_frames(dataset, frame_num=7):
    root = '..'
    data_dir = os.path.join(root, dataset, 'testing', 'top_depth') # 수정 #

    file_list = sorted(os.listdir(data_dir))
    frames_list = []
    videos_list = []

    total_frames = 0
    videos = 0
    start_ind = frame_num // 2

    for video_file in file_list:
        if video_file not in videos_list:
            videos_list.append(video_file)
        frame_list = os.listdir(data_dir + '/' + video_file)
        videos += 1
        length = len(frame_list)
        total_frames += length # 각 비디오 디렉토리의 모든 프레임 파일을 읽고, 총 프레임 수를 계산
        for frame in range(start_ind, length - start_ind):
            frames_list.append({"video_name": video_file, "frame": frame})

    print(f"Loaded {videos} videos with {total_frames} frames in total.")
    return frames_list

def remake_video_output(video_output, dataset='DAD_Jigsaw'): 
    # 주어진 video_output 데이터를 spatial과 temporal로 나누어 점수를 계산하고, 그 결과를 반환
    video_length = video_label_length(dataset=dataset) # video_label_length 함수를 사용하여 각 비디오의 프레임 길이를 가져옴
    return_output_spatial = []
    return_output_temporal = []
    return_output_complete = []
    video_l = sorted(list(video_output.keys())) # video 이름
    for i in range(len(video_l)):
        video = video_l[i]
        frame_record = video_output[video]
        frame_l = sorted(list(frame_record.keys())) # frame 이름
        video_ = np.ones(video_length[video]) 
        #비디오의 점수를 저장할 video_ 배열을 1로 초기화
        #각 비디오마다 점수를 저장할 배열을 생성
        video2_ = np.ones(video_length[video])

        local_max_ = 0
        local_max2_ = 0
        for fno in frame_l:
            clip_record = frame_record[fno]
            clip_record = np.array(clip_record)
            video_[fno], video2_[fno] = clip_record.min(0) # 각 spartial, temporal score
            # 각 프레임의 점수 clip_record를 읽고, min 함수를 사용하여 최소 점수를 계산해 video_ 배열에 저장

            local_max_ = max(clip_record[:, 0].max(), local_max_)
            local_max2_ = max(clip_record[:, 1].max(), local_max2_)

        # spatial
        non_ones = (video_ != 1).nonzero()[0]
        local_max_ = max(video_[non_ones])
        video_[non_ones] = (video_[non_ones] - min(video_)) / (local_max_ - min(video_))

        # temporal
        non_ones = (video2_ != 1).nonzero()[0]
        local_max2_ = max(video2_[non_ones])
        video2_[non_ones] = (video2_[non_ones] - min(video2_)) / (local_max2_ - min(video2_))

        # spatial
        video_ = score_smoothing(video_)
        # temporal
        video2_ = score_smoothing(video2_)

        # Store the results
        return_output_spatial.append(video_)
        return_output_temporal.append(video2_)

        # Combined spatial and temporal results
        combined_video = (video2_ + video_) / 2.0
        return_output_complete.append(combined_video)

    return return_output_spatial, return_output_temporal, return_output_complete


def evaluate_auc(video_output, dataset='DAD_Jigsaw'):#
    result_dict = {'dataset': dataset, 'psnr': video_output}
    smoothed_results, aver_smoothed_result = evaluate.evaluate_all(result_dict, reverse=True, smoothing=True) # 수정 
    print("(smoothing: True): {}  aver_result: {}".format(smoothed_results, aver_smoothed_result))
    return smoothed_results, aver_smoothed_result

#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Prediction')
    parser.add_argument('--file', default=None, type=str, help='pkl file')
    parser.add_argument('--dataset', default='ped2', type=str)
    parser.add_argument('--frame_num', required=True, type=int)

    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        output = pickle.load(f)

    # Process video outputs based on the dataset type
    video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(output, dataset=args.dataset)
    
    # Evaluate AUC for spatial, temporal, and combined outputs
    evaluate_auc(video_output_spatial, dataset=args.dataset)
    evaluate_auc(video_output_temporal, dataset=args.dataset)
    evaluate_auc(video_output_complete, dataset=args.dataset)

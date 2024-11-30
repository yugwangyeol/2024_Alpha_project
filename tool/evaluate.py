import numpy as np
import scipy.io as scio
import sys
from sklearn import metrics
import os
import argparse
import pickle
import math
import json

###
import matplotlib.pyplot as plt

class StatsTracker:
    def __init__(self):
        self.stats = []

    def update(self, epoch, psnr_records):
        epoch_stats = {
            'epoch': epoch,
            'min': np.min([np.min(v) for v in psnr_records]),
            'max': np.max([np.max(v) for v in psnr_records]),
            'mean': np.mean([np.mean(v) for v in psnr_records]),
            'std': np.mean([np.std(v) for v in psnr_records]),
            'nan_count': sum([np.isnan(v).sum() for v in psnr_records]),
            'inf_count': sum([np.isinf(v).sum() for v in psnr_records])
        }
        self.stats.append(epoch_stats)

    def print_summary(self):
        for stat in self.stats:
            print(f"Epoch {stat['epoch']}: Min={stat['min']:.2f}, Max={stat['max']:.2f}, "
                  f"Mean={stat['mean']:.2f}, Std={stat['std']:.2f}, "
                  f"NaN Count={stat['nan_count']}, Inf Count={stat['inf_count']}")

def log_epoch_stats(epoch, psnr_records):
    for i, video_data in enumerate(psnr_records):
        print(f"Epoch {epoch}, Video {i}:")
        print(f"  Min: {np.min(video_data)}")
        print(f"  Max: {np.max(video_data)}")
        print(f"  Mean: {np.mean(video_data)}")
        print(f"  Std: {np.std(video_data)}")
        print(f"  NaN count: {np.isnan(video_data).sum()}")
        print(f"  Inf count: {np.isinf(video_data).sum()}")

def visualize_data_distribution(epoch, psnr_records):
    plt.figure(figsize=(10, 6))
    for i, video_data in enumerate(psnr_records):
        plt.hist(video_data, bins=50, alpha=0.5, label=f'Video {i}')
    plt.title(f'Data Distribution at Epoch {epoch}')
    plt.xlabel('PSNR values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'./dist/distribution_epoch_{epoch}.png')
    plt.close()
###

# Path to the dataset directory
DATA_DIR = '../'

# Normalize scores in each sub video
NORMALIZE = True

# Number of history frames to ignore at the beginning
DECIDABLE_IDX = 3

def parser_args():
    parser = argparse.ArgumentParser(description='Evaluating the model and computing ROC/AUC.')
    parser.add_argument('-f', '--file', type=str, help='The path of the loss file.')
    parser.add_argument('-t', '--type', type=str, default='compute_auc',
                        help='The type of evaluation. Options: plot_roc, compute_auc, test_func. Default is compute_auc.')
    return parser.parse_args()

def score_smoothing(score, ws=25, function='mean', sigma=10):
    assert ws % 2 == 1, 'Window size must be odd'
    # ws: 윈도우 크기. 반드시 홀수여야 하며, 스무딩을 적용할 범위를 설정
    # weight: 주어진 윈도우 크기에 따라 각 점수에 가중치를 부여하기 위해 정의한 배열
    # 최종적으로 weight 배열의 합이 1이 되도록 정규화
    assert function in ['mean', 'gaussian'], 'Invalid window function type'

    r = ws // 2
    weight = np.ones(ws)
    for i in range(ws):
        if function == 'mean':
            weight[i] = 1. / ws
        elif function == 'gaussian':
            weight[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    weight /= weight.sum()
    new_score = score.copy()
    new_score[r: score.shape[0] - r] = np.correlate(score, weight, mode='valid')
    return new_score

def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter  

# Class to record evaluation results
class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return f'dataset = {self.dataset}, auc = {self.auc}'

# Class to load ground truth data for specific datasets
class GroundTruthLoader(object):
    DADJIGSAW = 'DAD_Jigsaw'
    DADJIGSAW_LABEL_PATH = os.path.join(DATA_DIR, 'DAD_Jigsaw/testing/frame_masks')

    def __init__(self):
        # Maps the dataset name to the corresponding label path
        self.mapping = {
            GroundTruthLoader.DADJIGSAW: GroundTruthLoader.DADJIGSAW_LABEL_PATH
        }

    def __call__(self, dataset):
        if dataset == GroundTruthLoader.DADJIGSAW:
            return self.__load_dadjigsaw_gt()

    @staticmethod
    def __load_dadjigsaw_gt():
        # Load ground truth for DADJIGSAW dataset
        video_path_list = os.listdir(GroundTruthLoader.DADJIGSAW_LABEL_PATH)
        video_path_list.sort()
        gt = [np.load(os.path.join(GroundTruthLoader.DADJIGSAW_LABEL_PATH, video)) for video in video_path_list]
        return gt

# Function to load PSNR values and corresponding ground truth labels
def load_psnr_gt(results):
    dataset = results['dataset']
    psnr_records = results['psnr']
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)
    assert len(psnr_records) == len(gt), f'Number of saved videos does not match ground truth: {len(psnr_records)} != {len(gt)}'
    return dataset, psnr_records, gt

# Function to compute AUC for the given results
def compute_auc(res, reverse, smoothing):
    dataset, psnr_records, gt = load_psnr_gt(res)
    num_videos = len(psnr_records)
    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)

    #tracker = StatsTracker()  # Initialize StatsTracker

    for i in range(num_videos):
        distance = psnr_records[i]
        
        # Log stats for each epoch
        #log_epoch_stats(i, [distance])
        
        # Update tracker
        #tracker.update(i, [distance])
        
        # Visualize data distribution
        #visualize_data_distribution(i, [distance])

        if np.isnan(distance).all() or np.isinf(distance).all():
            print(f"Skipping epoch {i} due to all NaN or Inf values in distance")
            continue 
        if np.isnan(distance).any() or np.isinf(distance).any():
            #print(distance)
            distance = np.nan_to_num(distance, nan=np.nanmean(distance))
            print(f"change nan epoch {i} : {np.nanmean(distance)}")
            continue 

        if NORMALIZE:
            min_val = distance.min()
            max_val = distance.max()
            if max_val > min_val:
                distance = (distance - min_val) / (max_val - min_val)
            else:
                distance = np.zeros_like(distance)
            if reverse:
                distance = 1 - distance
        if smoothing:
            filter_2d = gaussian_filter_(np.arange(1, 50), 20)
            padding_size = len(filter_2d) // 2
            in_ = np.concatenate((distance[:padding_size], distance, distance[-padding_size:]))
            distance = np.correlate(in_, filter_2d, 'valid')

        scores = np.concatenate((scores[:], distance), axis=0)
        labels = np.concatenate((labels[:], gt[i]), axis=0)

    # Print summary at the end
    #tracker.print_summary()

    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return RecordResult(fpr, tpr, auc, dataset)

# Function to compute the average AUC across all videos
def compute_auc_average(res, reverse, smoothing):
    dataset, psnr_records, gt = load_psnr_gt(res)
    num_videos = len(psnr_records)
    auc = 0

    for i in range(num_videos):
        distance = psnr_records[i]
        
        if np.isnan(distance).all() or np.isinf(distance).all():
            print(f"Skipping epoch {i} due to all NaN or Inf values in distance")
            continue 
        if np.isnan(distance).any() or np.isinf(distance).any():
            distance = np.nan_to_num(distance, nan=np.nanmean(distance))
            print(f"change nan epoch {i} : {np.nanmean(distance)}")
            continue

        if NORMALIZE and reverse:
            distance = 1 - distance
        if smoothing:
            filter_2d = gaussian_filter_(np.arange(1, 50), 20)
            padding_size = len(filter_2d) // 2
            in_ = np.concatenate((distance[:padding_size], distance, distance[-padding_size:]))
            distance = np.correlate(in_, filter_2d, 'valid')

        fpr, tpr, _ = metrics.roc_curve(np.concatenate(([0], gt[i], [1])), np.concatenate(([0], distance, [1])), pos_label=1)
        _auc = metrics.auc(fpr, tpr)
        auc += _auc

    auc /= num_videos
    return [auc]

# Function to evaluate all results and return AUC and average AUC
def evaluate_all(res, reverse=True, smoothing=True):
    result = compute_auc(res, reverse, smoothing)
    aver_result = compute_auc_average(res, reverse, smoothing)
    return result, aver_result

# Main entry point for the script
if __name__ == '__main__':
    pickle_path = './test.pkl'
    result = evaluate_all(pickle_path, reverse=True, smoothing=True)
    print(result)

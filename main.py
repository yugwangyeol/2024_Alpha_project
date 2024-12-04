import os
import argparse
import torch
import time
import pickle
import numpy as np

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VideoAnomalyDataset_C3D
from dataset_clip import VideoAnomalyDataset_Clip
from models import model

from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc
import wandb

torch.backends.cudnn.benchmark = False

wandb.init(project="Long-term JiasawVAD on DAD")

def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    parser.add_argument("--val_step", type=int, default=5000)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--static_threshold", type=float, default=0.3)
    parser.add_argument("--sample_num", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="DAD")
    parser.add_argument("--data_type", type=str, default='top_IR',
                        choices=['front_depth', 'front_IR', 'top_depth', 'top_IR'])
    parser.add_argument("--save_epoch", type=int, default=1)
    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    return args


def train(args):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    frame_data_dir = f"../{args.dataset}/training/{args.data_type}/frames"
    frame_dataset = VideoAnomalyDataset_C3D(
        frame_data_dir,
        frame_num=args.sample_num,
        static_threshold=args.static_threshold
    )
    frame_dataloader = DataLoader(
        frame_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    clip_dataset = VideoAnomalyDataset_Clip(
        frame_data_dir,
        sample_num=args.sample_num,
        num_clips=5,
        static_threshold=args.static_threshold
    )
    clip_dataloader = DataLoader(
        clip_dataset,
        batch_size=args.batch_size // 2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    net = model.WideBranchNet(frame_num=args.sample_num, clip_num=5)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        print('load ' + args.checkpoint)
        net.load_state_dict(state, strict=True)
        net.cuda()
        smoothed_auc, smoothed_auc_avg, _ = val(args, net)
        exit(0)

    net.cuda(args.device)
    net = net.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

    log_dir = './log/{}/'.format(running_date)
    writer = SummaryWriter(log_dir)

    t0 = time.time()
    global_step = 0
    max_acc = -1
    timestamp_in_max = None

    for epoch in range(args.epochs):
        clip_iterator = iter(clip_dataloader)

        for it, frame_data in enumerate(frame_dataloader):
            # Frame 처리
            frame_video, frame_clips, frame_temp_labels, frame_spat_labels, frame_flag = (
                frame_data['video'], frame_data['clip'],
                frame_data['label'], frame_data["trans_label"],
                frame_data["temporal"]
            )

            try:
                clip_data = next(clip_iterator)
            except StopIteration:
                clip_iterator = iter(clip_dataloader)
                clip_data = next(clip_iterator)

            # Frame level processing (task 분리)
            frame_clips = frame_clips.cuda(args.device, non_blocking=True)
            frame_temp_labels = frame_temp_labels.cuda(args.device, non_blocking=True)
            frame_spat_labels = frame_spat_labels.cuda(args.device, non_blocking=True)
            frame_flag = frame_flag.cuda(args.device, non_blocking=True)
            
            frame_temp_logits, frame_spat_logits = net(frame_clips, mode='frame')
            
            frame_temp_logits = frame_temp_logits[frame_flag].reshape(-1, args.sample_num)
            frame_temp_labels = frame_temp_labels[frame_flag].reshape(-1)
            frame_temp_loss = criterion(
                frame_temp_logits,
                frame_temp_labels
            )

            frame_spat_logits = frame_spat_logits[~frame_flag].reshape(-1, 9)
            frame_spat_labels = frame_spat_labels[~frame_flag].reshape(-1)
            frame_spat_loss = criterion(
                frame_spat_logits,
                frame_spat_labels
            )
            
            # Clip level processing (task 동시)
            clips = clip_data['clips'].cuda(args.device, non_blocking=True)
            clip_temp_labels = clip_data['label'].cuda(args.device, non_blocking=True)
            clip_spat_labels = clip_data['spatial_label'].cuda(args.device, non_blocking=True)

            clip_temp_logits, clip_spat_logits = net(clips, mode='clip')

            clip_temp_loss = criterion(
                clip_temp_logits.reshape(-1, clip_temp_logits.size(-1)),
                clip_temp_labels.reshape(-1)
            )
            clip_spat_loss = criterion(
                clip_spat_logits.reshape(-1, clip_spat_logits.size(-1)),
                clip_spat_labels.reshape(-1)
            )

            # Total loss
            loss = frame_temp_loss + frame_spat_loss + clip_temp_loss + clip_spat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (global_step + 1) % args.print_interval == 0:
                metrics = {
                    "train/total_loss": loss.item(),
                    "train/frame_temporal_loss": frame_temp_loss.item(),
                    "train/frame_spatial_loss": frame_spat_loss.item(),
                    "train/clip_temporal_loss": clip_temp_loss.item(),
                    "train/clip_spatial_loss": clip_spat_loss.item(),
                    "train/epoch": epoch,
                    "train/global_step": global_step
                }
                wandb.log(metrics)
                print("[Epoch {}] Step: {}, Loss: {:.6f} (FT: {:.6f}, FS: {:.6f}, CT: {:.6f}, CS: {:.6f}) Time: {:.6f}".format(
                    epoch, global_step, loss.item(), frame_temp_loss.item(), frame_spat_loss.item(),
                    clip_temp_loss.item(), clip_spat_loss.item(), time.time() - t0))
                t0 = time.time()

            global_step += 1

            if global_step % args.val_step == 0 and epoch >= 5:
                smoothed_auc, smoothed_auc_avg, temp_timestamp = val(args, net)
                wandb.log({
                    "val/smoothed_auc": smoothed_auc.auc,
                    "val/smoothed_auc_avg": smoothed_auc_avg,
                })

                if smoothed_auc.auc > max_acc:
                    max_acc = smoothed_auc.auc
                    timestamp_in_max = temp_timestamp
                    save = './checkpoint/{}_{}.pth'.format('best', running_date)
                    torch.save(net.state_dict(), save)

                print('Current max: {} in {}'.format(max_acc, timestamp_in_max))
                net = net.train()

        if epoch % args.save_epoch == 0 and epoch >= 5:
            save = './save_ckpt/{}_{}.pth'.format(running_date, epoch)
            torch.save(net.state_dict(), save)
        
    wandb.finish()


def val(args, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    data_dir = os.path.join("../", args.dataset, 'testing', args.data_type, 'frames')

    # Frame-level testing dataset
    frame_dataset = VideoAnomalyDataset_C3D(data_dir, 
                                            frame_num=args.sample_num)
    frame_dataloader = DataLoader(frame_dataset, 
                                batch_size=256, 
                                shuffle=False, 
                                num_workers=4, 
                                drop_last=False)

    # Clip-level testing dataset
    clip_dataset = VideoAnomalyDataset_Clip(data_dir, 
                                            sample_num=args.sample_num,
                                            num_clips=5)
    clip_dataloader = DataLoader(clip_dataset, 
                            batch_size=256, 
                            shuffle=False, 
                            num_workers=4, 
                            drop_last=False)

    net.eval()

    # Store outputs
    frame_video_output = {}
    clip_video_output = {}

    # Frame-level prediction
    print("Processing frame-level predictions...")
    for idx, data in enumerate(tqdm(frame_dataloader)):
        videos = data["video"]
        frames = data["frame"].tolist()
        clip = data["clip"].cuda(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = net(clip, mode='frame')
            if torch.isnan(temp_logits).any():
                print(f"NaN detected in frame output at batch {idx}")
                print(f"Input statistics: min={data['clip'].min()}, max={data['clip'].max()}")

            temp_logits = temp_logits.view(-1, args.sample_num, args.sample_num)
            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()
        
        for video_, frame_, s_score_, t_score_  in zip(videos, frames, scores, scores2):
            if video_ not in frame_video_output:
                frame_video_output[video_] = {}
            if frame_ not in frame_video_output[video_]:
                frame_video_output[video_][frame_] = []
            frame_video_output[video_][frame_].append([s_score_, t_score_])

    # Clip-level prediction
    print("Processing clip-level predictions...")
    for idx, data in enumerate(tqdm(clip_dataloader)):
        videos = data["video"]
        center_frames = data["center_frame"].tolist()
        clips = data["clips"].cuda(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = net(clips, mode='clip')
            if torch.isnan(temp_logits).any():
                print(f"NaN detected in clip output at batch {idx}")
                print(f"Input statistics: min={data['clip'].min()}, max={data['clip'].max()}")

            temp_logits = temp_logits.view(-1, 5, 5)
            spat_logits = spat_logits.view(-1, 9, 9)

        temperature = 2.0
        spat_probs = F.softmax(spat_logits / temperature, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits / temperature, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()
        
        for video_, frame_, s_score_, t_score_  in zip(videos, center_frames, scores, scores2):
            if video_ not in clip_video_output:
                clip_video_output[video_] = {}
            if frame_ not in clip_video_output[video_]:
                clip_video_output[video_][frame_] = []
            clip_video_output[video_][frame_].append([s_score_, t_score_])
            
    # print(f"clip_video_output: {clip_video_output}")

    # Combine frame and clip predictions
    combined_video_output = {}
    for video_ in set(frame_video_output.keys()) | set(clip_video_output.keys()):
        combined_video_output[video_] = {}
        all_frames = set(frame_video_output.get(video_, {}).keys()) | set(clip_video_output.get(video_, {}).keys())
        
        for frame_ in all_frames:
            frame_scores = frame_video_output.get(video_, {}).get(frame_, [[1.0, 1.0]])
            clip_scores = clip_video_output.get(video_, {}).get(frame_, [[1.0, 1.0]])
            
            # Combine scores (average of frame and clip predictions)
            combined_s_score = (frame_scores[0][0] + clip_scores[0][0]) / 2
            combined_t_score = (frame_scores[0][1] + clip_scores[0][1]) / 2
            
            combined_video_output[video_][frame_] = [[combined_s_score, combined_t_score]]

    # Save predictions
    pickle_path = './log/video_output_ori_{}.pkl'.format(running_date)
    with open(pickle_path, 'wb') as write:
        pickle.dump(combined_video_output, write, pickle.HIGHEST_PROTOCOL)

    # Evaluate predictions
    frame_spatial, frame_temporal, frame_complete = remake_video_output(frame_video_output)
    frame_auc, frame_auc_list = evaluate_auc(frame_complete)
    print(f"Frame-level AUC: {frame_auc}")

    clip_spatial, clip_temporal, clip_complete = remake_video_output(clip_video_output)
    clip_auc, clip_auc_list = evaluate_auc(clip_complete)
    print(f"Clip-level AUC: {clip_auc}")

    combined_spatial, combined_temporal, combined_complete = remake_video_output(combined_video_output)
    combined_auc, combined_auc_list = evaluate_auc(combined_complete)
    print(f"Combined AUC: {combined_auc}")

    return combined_auc, np.mean(combined_auc_list), running_date

if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)

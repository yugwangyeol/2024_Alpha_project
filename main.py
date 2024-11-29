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

from dataset import VideoAnomalyDataset_C3D_Frame, VideoAnomalyDataset_C3D_Clip
from models.model import WideBranchNet

from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc

torch.backends.cudnn.benchmark = False

def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    parser.add_argument("--val_step", type=int, default=500)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--static_threshold", type=float, default=0.3)
    parser.add_argument("--frame_num", type=int, default=7)
    parser.add_argument("--clip_num", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="DAD_Jigsaw")
    parser.add_argument("--data_type", type=str, default='top_depth', 
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
    data_dir = os.path.join("../", args.dataset, 'training', args.data_type, 'frames')

    frame_dataset = VideoAnomalyDataset_C3D_Frame(data_dir, 
                                                 frame_num=args.frame_num,
                                                 static_threshold=args.static_threshold)

    clip_dataset = VideoAnomalyDataset_C3D_Clip(data_dir, 
                                               frame_num=args.frame_num,
                                               clip_num=args.clip_num,
                                               static_threshold=args.static_threshold)

    frame_dataloader = DataLoader(frame_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=4, 
                                pin_memory=True)

    clip_dataloader = DataLoader(clip_dataset, 
                               batch_size=args.batch_size, 
                               shuffle=True, 
                               num_workers=4, 
                               pin_memory=True)

    # Model
    net = WideBranchNet(frame_num=args.frame_num, clip_num=args.clip_num)
    
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        net.load_state_dict(state, strict=True)
        net.cuda()
        smoothed_auc, smoothed_auc_avg, _ = val(args, net)
        exit(0)

    net.cuda(args.device)
    net = net.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

    # Train
    log_dir = './log/{}/'.format(running_date)
    writer = SummaryWriter(log_dir)
    
    t0 = time.time()
    global_step = 0
    max_acc = -1
    timestamp_in_max = None

    for epoch in range(args.epochs):
        frame_iter = iter(frame_dataloader)
        clip_iter = iter(clip_dataloader)

        while True:
            try:
                clip_data = next(clip_iter)
            except StopIteration:
                clip_iter = iter(clip_dataloader)
                clip_data = next(clip_iter)
                
            # Frame predictions (clip_num times)
            frame_losses = []
            for _ in range(args.clip_num):
                try:
                    frame_data = next(frame_iter)
                except StopIteration:
                    frame_iter = iter(frame_dataloader)
                    frame_data = next(frame_iter)

                # Process frame data
                f_video, f_clip, f_temp_labels, f_spat_labels, f_flag = (
                    frame_data['video'], frame_data['clip'],
                    frame_data['label'], frame_data["trans_label"],
                    frame_data["temporal"]
                )

                f_clip = f_clip.cuda(args.device, non_blocking=True)
                f_temp_labels = f_temp_labels[f_flag].long().view(-1).cuda(args.device)
                f_spat_labels = f_spat_labels[~f_flag].long().view(-1).cuda(args.device)

                f_temp_logits, f_spat_logits = net(f_clip, mode='frame')
                f_temp_loss = criterion(f_temp_logits, f_temp_labels)
                f_spat_loss = criterion(f_spat_logits, f_spat_labels)
                frame_losses.append(f_temp_loss + f_spat_loss)

            # Process clip data
            c_video, c_clip, c_temp_labels, c_spat_labels, c_flag = (
                clip_data['video'], clip_data['clip'],
                clip_data['label'], clip_data["trans_label"],
                clip_data["temporal"]
            )

            c_clip = c_clip.cuda(args.device, non_blocking=True)
            c_temp_labels = c_temp_labels[c_flag].long().view(-1).cuda(args.device)
            c_spat_labels = c_spat_labels[~c_flag].long().view(-1).cuda(args.device)

            c_temp_logits, c_spat_logits = net(c_clip, mode='clip')
            c_temp_loss = criterion(c_temp_logits, c_temp_labels)
            c_spat_loss = criterion(c_spat_logits, c_spat_labels)

            # Total loss
            total_loss = sum(frame_losses) + c_temp_loss + c_spat_loss

            # Optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            writer.add_scalar('Train/Loss', total_loss.item(), global_step=global_step)
            
            if global_step % args.print_interval == 0:
                print("[{}] Step: {}, Loss: {:.6f}, Time: {:.6f}".format(
                    epoch, global_step, total_loss.item(), time.time() - t0))
                t0 = time.time()

            global_step += 1

            # Validation
            if global_step % args.val_step == 0 and epoch >= 5:
                smoothed_auc, smoothed_auc_avg, temp_timestamp = val(args, net)
                writer.add_scalar('Test/smoothed_auc', smoothed_auc, global_step=global_step)
                writer.add_scalar('Test/smoothed_auc_avg', smoothed_auc_avg, global_step=global_step)

                if smoothed_auc > max_acc:
                    max_acc = smoothed_auc
                    timestamp_in_max = temp_timestamp
                    save = './checkpoint/{}_{}.pth'.format('best', running_date)
                    torch.save(net.state_dict(), save)

                print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max)
                net = net.train()
            
            # Check if we should end the epoch (based on frame_dataloader length)
            if global_step % len(frame_dataloader) == 0:
                break

        if epoch % args.save_epoch == 0 and epoch >= 5:
            save = './save_ckpt/{}_{}.pth'.format(running_date, epoch)
            torch.save(net.state_dict(), save)

def val(args, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    data_dir = os.path.join("../", args.dataset, 'testing', args.data_type, 'frames')

    # Frame-level testing dataset
    frame_dataset = VideoAnomalyDataset_C3D_Frame(data_dir, 
                                                 frame_num=args.frame_num)
    frame_dataloader = DataLoader(frame_dataset, 
                                batch_size=256, 
                                shuffle=False, 
                                num_workers=4, 
                                drop_last=False)

    # Clip-level testing dataset
    clip_dataset = VideoAnomalyDataset_C3D_Clip(data_dir, 
                                               frame_num=args.frame_num,
                                               clip_num=args.clip_num)
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

            temp_logits = temp_logits.view(-1, args.frame_num, args.frame_num)
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
        frames = data["frame"].tolist()
        clip = data["clip"].cuda(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = net(clip, mode='clip')
            if torch.isnan(temp_logits).any():
                print(f"NaN detected in clip output at batch {idx}")
                print(f"Input statistics: min={data['clip'].min()}, max={data['clip'].max()}")

            temp_logits = temp_logits.view(-1, args.clip_num, args.clip_num)
            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        scores2 = diag2.min(-1)[0].cpu().numpy()
        
        for video_, frame_, s_score_, t_score_  in zip(videos, frames, scores, scores2):
            if video_ not in clip_video_output:
                clip_video_output[video_] = {}
            if frame_ not in clip_video_output[video_]:
                clip_video_output[video_][frame_] = []
            clip_video_output[video_][frame_].append([s_score_, t_score_])

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